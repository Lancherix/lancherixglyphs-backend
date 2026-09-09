"""
main.py
=======

Backend FastAPI para Lancherix Visual Code. Envuelve, sin modificar
su logica interna:

  - lancherix_generator.py      (texto -> imagen PNG)
  - lancherix_camera_reader.py  (foto  -> texto, via corner markers)

Endpoints:

  POST /generate
      body JSON: {"text": "...", "ecc": 4 (opcional)}
      -> devuelve el PNG generado (image/png)

  POST /decode
      multipart/form-data: file=<imagen> , ecc=<int opcional>
      -> devuelve JSON: {"text": "...", "images": [...], "logs": "..."}
         o {"error": "...", "images": [...], "logs": "..."}

  GET /health
      -> {"status": "ok"}  (util para el health check de Render)

Cada request corre en su propio directorio temporal, asi los archivos
de debug que generan lancherix_camera_reader.py / lancherix_generator.py
(png intermedios) no chocan entre pedidos concurrentes ni se acumulan
en el filesystem del backend.

Los prints de lancherix_camera_reader.py (incluyendo los timings
"[timing] ..." agregados para diagnosticar rendimiento) se capturan
via redirect_stdout durante la decodificacion y se devuelven en el
campo "logs" de la respuesta -- tanto si la decodificacion tiene
exito como si falla, para poder ver hasta donde llego el pipeline.
"""

from __future__ import annotations

import base64
import asyncio
import contextlib
import io
import shutil
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from lancherix_generator import render_png
from lancherix_camera_reader import read_camera_image_via_markers
from lancherix_shapes import DEFAULT_ECC_SYMBOLS

app = FastAPI(title="Lancherix Visual Code API")

# CORS abierto: ajusta allow_origins a tu dominio de frontend en
# produccion (por ejemplo, el dominio donde quede publicada la app de
# React) en vez de "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# read_camera_image_via_markers() escribe sus PNGs de debug relativos
# al directorio de trabajo actual (cwd), no a la carpeta del archivo
# de entrada. os.chdir() es un estado GLOBAL del proceso, asi que dos
# decodificaciones concurrentes pisarian el cwd una de la otra. Este
# lock serializa /decode para que eso nunca pase (el pipeline de
# vision por si solo ya toma un momento, asi que no es una perdida de
# rendimiento grave para un backend de uso personal).
_decode_lock = asyncio.Lock()


class _DecodeFailure(Exception):
    """
    Envuelve cualquier excepcion lanzada dentro del pipeline de
    decodificacion junto con los logs (stdout) capturados hasta el
    momento del fallo, para que el endpoint pueda devolver ambos --
    el error Y el proceso que se llego a completar -- en vez de
    perder los logs cuando algo sale mal.
    """
    def __init__(self, original: Exception, logs: str):
        super().__init__(str(original))
        self.original = original
        self.logs = logs


def _encode_debug_images(workdir: Path, filenames: list[str]) -> list[dict]:
    """
    Lee cada PNG de debug generado (relativo a workdir) y lo devuelve
    como {"name": ..., "data": "data:image/png;base64,..."} listo
    para mandar en la respuesta JSON. Si algun archivo no llego a
    generarse (por ejemplo, el error ocurrio antes de esa fase), se
    lo salta en silencio.
    """
    images = []
    for name in filenames:
        path = workdir / name
        if not path.exists():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        images.append({"name": name, "data": f"data:image/png;base64,{encoded}"})
    return images


def _run_decode_sync(
    workdir: Path, input_name: str, ecc: int, debug_images_out: list,
) -> tuple[str, str]:
    """
    Corre read_camera_image_via_markers() en un hilo, con el cwd
    apuntando a `workdir` (necesario porque esa funcion escribe sus
    PNGs de debug relativos al cwd) y capturando todo lo que imprime
    por stdout (incluyendo los prints "[timing] ..." de diagnostico
    de rendimiento) en un buffer, que se devuelve junto al texto
    decodificado.

    Si algo falla, se relanza como _DecodeFailure con los logs
    capturados hasta ese punto adjuntos, para no perderlos.
    """
    import os

    cwd_before = os.getcwd()
    os.chdir(workdir)
    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer):
            text = read_camera_image_via_markers(
                input_name, num_ecc_symbols=ecc, debug_images_out=debug_images_out,
            )
        return text, log_buffer.getvalue()
    except Exception as exc:
        raise _DecodeFailure(exc, log_buffer.getvalue()) from exc
    finally:
        os.chdir(cwd_before)


@app.post("/generate")
def generate(payload: dict):
    """
    Genera un Lancherix Visual Code para el texto recibido y devuelve
    el PNG resultante.

    body JSON esperado: {"text": "...", "ecc": 4 (opcional)}
    """
    text = payload.get("text")
    if not text or not str(text).strip():
        raise HTTPException(status_code=400, detail="El campo 'text' es requerido.")

    ecc = payload.get("ecc", DEFAULT_ECC_SYMBOLS)
    try:
        ecc = int(ecc)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="'ecc' debe ser un entero.")

    workdir = Path(tempfile.mkdtemp(prefix="lancherix_gen_"))
    output_path = workdir / "lancherix_code.png"

    try:
        render_png(str(text), str(output_path), num_ecc_symbols=ecc)
    except Exception as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo generar el codigo: {exc}",
        )

    if not output_path.exists():
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="El archivo no se genero.")

    # FileResponse lee el archivo antes de que termine el request;
    # limpiamos el directorio temporal despues con un background task
    # simple usando el propio starlette BackgroundTask.
    from starlette.background import BackgroundTask

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename="lancherix_code.png",
        background=BackgroundTask(shutil.rmtree, str(workdir), ignore_errors=True),
    )


@app.post("/decode")
async def decode(file: UploadFile = File(...), ecc: int = Form(DEFAULT_ECC_SYMBOLS)):
    """
    Recibe una foto, la decodifica via read_camera_image_via_markers(),
    y devuelve JSON con:
      - "text": el texto decodificado (solo si tuvo exito)
      - "error": el mensaje de error (solo si fallo)
      - "images": las fotos de debug generadas por el pipeline (se
        incluyen aunque haya habido un error, hasta donde se haya
        llegado)
      - "logs": todo lo que el pipeline imprimio por stdout durante
        la decodificacion (incluyendo los timings "[timing] ..."),
        tanto si tuvo exito como si fallo
    """
    workdir = Path(tempfile.mkdtemp(prefix="lancherix_decode_"))
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    input_path = workdir / f"input{suffix}"
    debug_image_names: list[str] = []

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        async with _decode_lock:
            text, logs = await asyncio.to_thread(
                _run_decode_sync, workdir, input_path.name, ecc, debug_image_names,
            )

        images = _encode_debug_images(workdir, debug_image_names)
        return JSONResponse({"text": text, "images": images, "logs": logs})

    except _DecodeFailure as fail:
        images = _encode_debug_images(workdir, debug_image_names)

        if isinstance(fail.original, ValueError):
            return JSONResponse(
                {"error": str(fail.original), "images": images, "logs": fail.logs},
                status_code=422,
            )

        traceback.print_exc()
        return JSONResponse(
            {
                "error": f"Error interno al decodificar: {fail.original}",
                "images": images,
                "logs": fail.logs,
            },
            status_code=500,
        )

    except Exception as exc:
        # Fallo fuera del pipeline de decodificacion en si (por
        # ejemplo, al guardar el archivo subido) -- no hay logs de
        # _run_decode_sync que adjuntar en este caso.
        traceback.print_exc()
        images = _encode_debug_images(workdir, debug_image_names)
        return JSONResponse(
            {"error": f"Error interno al decodificar: {exc}", "images": images},
            status_code=500,
        )

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)