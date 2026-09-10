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

--------------------------------------------------------------------
NOTA IMPORTANTE sobre /decode y concurrencia (leer antes de tocar esto)
--------------------------------------------------------------------
read_camera_image_via_markers() escribe sus PNGs de debug relativos
al directorio de trabajo actual (cwd) del proceso, no a una ruta
absoluta. Eso significa que necesitamos aislar el cwd por request.

La version anterior de este archivo resolvia esto con os.chdir() +
un asyncio.Lock() global que serializaba TODOS los /decode entre si
(porque os.chdir() es un estado global del PROCESO -- si dos threads
del mismo proceso lo cambiaran en paralelo, se pisarian entre si).

El problema de esa solucion: convertia cualquier decodificacion lenta
en deuda de cola para TODAS las que llegaran despues, sin importar
cuan simples fueran. Con varias requests encoladas, el tiempo de
respuesta que ve el cliente pasa a ser "tiempo de cola + tiempo real
de proceso", y el tiempo de cola crece sin limite si siguen llegando
requests mientras la cola no se vacia -- lo que explica el sintoma de
"las faciles tambien empiezan a demorar despues de una dificil".

La solucion de raiz: en vez de compartir un unico proceso (con su
unico cwd global) entre threads, cada decodificacion corre en su
propio PROCESO hijo (ProcessPoolExecutor). Cada proceso tiene su
propio cwd independiente desde que arranca, asi que no hace falta
ningun lock -- varias decodificaciones pueden correr en paralelo de
verdad, acotadas por la cantidad de workers del pool (ver
DECODE_POOL_WORKERS mas abajo) en vez de por "una a la vez".
"""

from __future__ import annotations

import base64
import asyncio
import contextlib
import io
import multiprocessing
import os
import shutil
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
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


# --------------------------------------------------------------------
# Pool de procesos para /decode (ver nota arriba)
# --------------------------------------------------------------------

# Cuantas decodificaciones corren en paralelo como maximo. Cada worker
# es un proceso separado que mantiene cargado cv2/numpy/etc. una vez
# que procesa su primera tarea, asi que hay un costo de memoria fijo
# por worker (aparte del pico durante la decodificacion en si). En una
# instancia chica de Render (poca RAM, 1-2 vCPU compartidas) conviene
# no poner esto igual a multiprocessing.cpu_count() a ciegas -- 2 o 3
# suele ser un piso razonable para empezar; subilo si ves que el pool
# se queda corto (requests encoladas con CPU disponible) y bajalo si
# ves swapping/OOM.
DECODE_POOL_WORKERS = min(4, multiprocessing.cpu_count())

_process_pool: ProcessPoolExecutor | None = None


def _warmup_noop() -> None:
    """Trivial task submitted to each pool worker on startup just to
    force it to spawn/fork now, instead of on the first real request."""
    return None


def _get_process_pool() -> ProcessPoolExecutor:
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=DECODE_POOL_WORKERS)
    return _process_pool


@app.on_event("startup")
def _warm_process_pool():
    pool = _get_process_pool()
    futures = [pool.submit(_warmup_noop) for _ in range(DECODE_POOL_WORKERS)]
    for f in futures:
        f.result()


def _decode_worker(workdir_str: str, input_name: str, ecc: int) -> dict:
    """
    Corre en un PROCESO hijo separado (no un thread), asi que su cwd es
    independiente del proceso principal y de cualquier otro worker --
    a diferencia de la version con threads + os.chdir(), aca no hace
    falta ningun lock: cada llamada tiene su propio cwd desde que el
    proceso arranca, sin riesgo de que dos decodificaciones se pisen.

    Devuelve siempre un dict (nunca lanza para fallos ESPERADOS del
    pipeline, como un ValueError de decodificacion) para no depender
    de que las excepciones se puedan pickle/despicklear correctamente
    al cruzar el limite entre procesos -- lo cual no esta garantizado
    para tipos de excepcion arbitrarios definidos en otros modulos.

    Formato del dict:
      exito:  {"ok": True, "text": ..., "logs": ..., "images": [...]}
      fallo:  {"ok": False, "error_type": ..., "error_msg": ...,
               "logs": ..., "images": [...]}
    """
    os.chdir(workdir_str)
    debug_image_names: list[str] = []
    log_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(log_buffer):
            text = read_camera_image_via_markers(
                input_name, num_ecc_symbols=ecc, debug_images_out=debug_image_names,
            )
        return {
            "ok": True,
            "text": text,
            "logs": log_buffer.getvalue(),
            "images": debug_image_names,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "logs": log_buffer.getvalue(),
            "images": debug_image_names,
        }


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
    Recibe una foto, la decodifica via read_camera_image_via_markers()
    (corriendo en un proceso separado del pool -- ver _decode_worker),
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

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _get_process_pool(), _decode_worker, str(workdir), input_path.name, ecc,
        )

        images = _encode_debug_images(workdir, result["images"])

        if result["ok"]:
            return JSONResponse(
                {"text": result["text"], "images": images, "logs": result["logs"]}
            )

        if result["error_type"] == "ValueError":
            return JSONResponse(
                {
                    "error": result["error_msg"],
                    "images": images,
                    "logs": result["logs"],
                },
                status_code=422,
            )

        print(
            f"[decode] error interno ({result['error_type']}): {result['error_msg']}"
        )
        return JSONResponse(
            {
                "error": f"Error interno al decodificar: {result['error_msg']}",
                "images": images,
                "logs": result["logs"],
            },
            status_code=500,
        )

    except Exception as exc:
        # Fallo fuera del pipeline de decodificacion en si (por
        # ejemplo, al guardar el archivo subido, o al comunicarse con
        # el proceso del pool) -- no hay "logs" del pipeline interno
        # que adjuntar en este caso.
        traceback.print_exc()
        images = _encode_debug_images(workdir, [])
        return JSONResponse(
            {"error": f"Error interno al decodificar: {exc}", "images": images},
            status_code=500,
        )

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)