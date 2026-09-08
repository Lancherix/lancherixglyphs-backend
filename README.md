# Lancherix Backend

FastAPI wrapper around your existing Lancherix Visual Code pipeline
(`lancherix_generator.py`, `lancherix_camera_reader.py`,
`lancherix_corner_rectifier_4markers.py`, `lancherix_shapes.py`,
`lancherix_scanner_preprocessor.py`). None of the pipeline logic was
changed — `main.py` just calls `render_png(...)` and
`read_camera_image_via_markers(...)` directly.

## Endpoints

- `GET /health` -> `{"status": "ok"}`
- `POST /generate` — JSON body `{"text": "...", "ecc": 4}` (`ecc` optional)
  -> returns the PNG (`image/png`)
- `POST /decode` — multipart form: `file=<image>`, `ecc=<int>` (optional, form field)
  -> returns `{"text": "..."}` or `{"error": "..."}` (HTTP 422) if the
  code couldn't be located/decoded

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "hola"}' \
  -o test.png

curl -X POST http://localhost:8000/decode \
  -F "file=@test.png"
```

## Deploy to Render

1. Push this folder to a GitHub repo (or a subfolder of your existing
   repo — if it's a subfolder, set "Root Directory" in Render's
   dashboard to that path).
2. In Render: **New +** -> **Blueprint**, point it at the repo. It
   will pick up `render.yaml` automatically. Or: **New +** -> **Web
   Service**, and set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Once deployed you'll get a URL like
   `https://lancherix-backend.onrender.com`.
4. **Note (free tier):** Render's free web services spin down after
   ~15 min of inactivity; the first request after idling can take
   30-60s to wake up. That's normal, not a bug in your app.
5. Lock down CORS in `main.py` (`allow_origins=["*"]`) to your actual
   frontend domain once you know it.

## Wiring up the React app

See `Generator.jsx` and `Reader.jsx` in this same delivery for a
working example: they call `${API_BASE_URL}/generate` and
`${API_BASE_URL}/decode` respectively. Set `API_BASE_URL` (e.g. via a
`VITE_API_BASE_URL` env var) to your Render URL once deployed, or to
`http://localhost:8000` for local dev.
