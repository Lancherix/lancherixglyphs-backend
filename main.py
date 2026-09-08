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
      -> devuelve JSON: {"text": "..."} o {"error": "..."}

  GET /health
      -> {"status": "ok"}  (util para el health check de Render)

Cada request corre en su propio directorio temporal, asi los archivos
de debug que generan lancherix_camera_reader.py / lancherix_generator.py
(png intermedios) no chocan entre pedidos concurrentes ni se acumulan
en el filesystem del backend.
"""

from __future__ import annotations

import asyncio
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


def _run_decode_sync(workdir: Path, input_name: str, ecc: int) -> str:
    import os

    cwd_before = os.getcwd()
    os.chdir(workdir)
    try:
        return read_camera_image_via_markers(input_name, num_ecc_symbols=ecc)
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
    Decodifica un Lancherix Visual Code a partir de una foto (con
    perspectiva real arbitraria), usando los 4 corner markers.

    multipart/form-data:
        file : la imagen (jpg/png) capturada por la camara o subida
        ecc  : (opcional) num_ecc_symbols, debe coincidir con el usado
               al generar el codigo. Por defecto DEFAULT_ECC_SYMBOLS.
    """
    workdir = Path(tempfile.mkdtemp(prefix="lancherix_decode_"))
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    input_path = workdir / f"input{suffix}"

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # read_camera_image_via_markers escribe varios PNGs de debug
        # junto al archivo de entrada (mismo stem); al vivir en
        # `workdir` (temporal, por-request) no molestan ni persisten.
        # Serializado por _decode_lock (ver nota arriba) y corrido en
        # un thread aparte para no bloquear el event loop de asyncio.
        async with _decode_lock:
            text = await asyncio.to_thread(
                _run_decode_sync, workdir, input_path.name, ecc,
            )

        return JSONResponse({"text": text})

    except ValueError as exc:
        # Errores esperados del pipeline (no se detectaron los
        # markers, no se pudo decodificar, checksum invalido, etc).
        return JSONResponse({"error": str(exc)}, status_code=422)

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            {"error": f"Error interno al decodificar: {exc}"}, status_code=500,
        )

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)