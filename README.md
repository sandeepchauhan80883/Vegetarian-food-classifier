# VegSnap -- Web Frontend + API Backend

A small full-stack demo around the trained vegetarian-food classifier:
a FastAPI backend that loads the trained model and serves predictions,
and a plain HTML/CSS/JS frontend to upload a photo and see the result.

```
webapp/
  backend/
    app.py             # FastAPI server: /health, /predict
    requirements.txt
  frontend/
    index.html          # open this in a browser
  veg_classifier/         # same model/pipeline code as the main project
  artifacts/                # put your trained best_model.pt here
```

## 1. Get a trained checkpoint

Train the model using the main project notebook (ideally on the real
dataset, in Colab with a GPU -- see the top-level README.md). That run
saves `artifacts/best_model.pt`. Download that file and place it at:

```
webapp/artifacts/best_model.pt
```

(create the `artifacts` folder if it isn't there yet)

## 2. Start the backend

```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Check it's alive: open http://localhost:8000/health in a browser -- you
should see `"checkpoint_found": true`.

## 3. Open the frontend

Just double-click `webapp/frontend/index.html` to open it in your
browser (no build step, no server needed for the frontend itself). If
the backend address field doesn't already say `http://localhost:8000`,
type that in.

Upload a food photo and click **Classify photo**.

## Notes

- **CORS** is wide open (`allow_origins=["*"]`) in `backend/app.py` so the
  frontend can call it from a `file://` page during development / a
  project demo. Restrict this before deploying anywhere public.
- **Running the backend from Colab instead of your laptop:** Colab
  notebooks don't expose `localhost` to your own browser directly. Either
  (a) download `best_model.pt` and run the backend locally as above
  (recommended, and simplest for a demo), or (b) use a tunnel tool like
  `ngrok`/`pyngrok` inside the Colab notebook to get a public URL for the
  API, then paste that URL into the frontend's backend-address field.
- **Model not found error / 503 from `/predict`:** means
  `webapp/artifacts/best_model.pt` isn't there yet -- see step 1.
