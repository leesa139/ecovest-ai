# EcoVest AI – ESG Investment Advisor (FastAPI + web UI)

## Run in VS Code
```
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://localhost:8000 (API docs at /docs). Register an account, then use the app.

## Push to GitHub
Create an empty repo `ecovest-ai` on github.com, then:
```
git init
git add .
git commit -m "EcoVest AI"
git branch -M main
git remote add origin https://github.com/<your-username>/ecovest-ai.git
git push -u origin main
```

## Deploy on Render
1. render.com -> New -> Blueprint -> connect the GitHub repo (it reads render.yaml).
   Manual Web Service: Build `pip install -r requirements.txt`, Start `uvicorn main:app --host 0.0.0.0 --port $PORT`.
2. When the build finishes, the site is live at https://ecovest-ai.onrender.com

Note: the free plan sleeps when idle (first load ~50 s) and the SQLite data (users, history, watchlist) resets on redeploy. For permanent users add a Render persistent disk or Postgres.
