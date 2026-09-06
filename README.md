# SatQuery AI (Unified Deployment Edition)

Agentic multi-modal remote sensing vision-language assistant for analyzing single and paired satellite imagery through natural-language queries.

---

## Architecture: Unified Single-Service Hosting

This build is configured as a **single unified service**:
- The **FastAPI backend** directly serves the **HTML5/JS frontend** from `/` and `/app`.
- The frontend dynamically communicates with the backend on the same origin (no cross-origin URLs, no CORS setup, and no separate hosting services required).
- Specialist remote sensing model (`rs_classifier.pt`) is automatically fetched from Hugging Face (`KowhickMaran/rs-eurosat-classifier`) on demand.

---

## Option 1: Deploy for 100% FREE on Hugging Face Spaces (Recommended)

> **Why Hugging Face Spaces?**
> - **16 GB RAM + 2 vCPUs completely free** (no credit card needed).
> - Zero risk of Out-Of-Memory (OOM) crashes when PyTorch loads the CLIP model.
> - Fast model download since the checkpoint is already on Hugging Face.

### Steps:
1. Go to [huggingface.co](https://huggingface.co) and sign in.
2. Click **New Space** (or go to `https://huggingface.co/new-space`).
3. Set your Space settings:
   - **Space Name**: `satquery-ai` (or any name you prefer)
   - **Space SDK**: Choose **Docker** -> **Blank**
   - **Space Hardware**: Free (CPU basic · 2 vCPU · 16 GB · Free)
4. In your newly created Space, go to **Settings** -> **Variables and secrets**:
   - Add a new **Secret**:
     - Key: `GEMINI_API_KEY`
     - Value: `<Your Google Gemini API Key>`
5. Upload or push this `maran` folder to your Space repository:
   ```bash
   cd C:\Users\kowsh\Desktop\maran
   git init
   git add .
   git commit -m "Deploy SatQuery AI unified application"
   git branch -M main
   git remote add origin https://huggingface.co/spaces/<your-username>/satquery-ai
   git push -u origin main --force
   ```
6. Hugging Face will automatically build the `Dockerfile` and launch your app.
7. Access your live application at:
   `https://<your-username>-satquery-ai.hf.space`

---

## Option 2: Deploy to Railway (Single Unified Service)

1. Push this `maran` folder to a GitHub repository:
   ```bash
   cd C:\Users\kowsh\Desktop\maran
   git init
   git add .
   git commit -m "Initial commit for unified deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/satquery-ai.git
   git push -u origin main
   ```
2. On [Railway](https://railway.app):
   - Click **New Project** -> **Deploy from GitHub repo**.
   - Select your repository.
   - Go to the **Variables** tab and set:
     - `GEMINI_API_KEY`: Your Gemini API key.
     - `GEMINI_MODEL`: `gemini-1.5-flash` (optional, defaults to `gemini-1.5-flash`).
     - `HF_REPO_ID`: `KowhickMaran/rs-eurosat-classifier` (optional).
   - Under **Settings** -> **Networking**, click **Generate Domain**.
3. Access your web app directly at your Railway domain (e.g., `https://your-service.up.railway.app`).

---

## Local Development / Testing

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create `.env` from `.env.example` and set your key:
   ```bash
   cp .env.example .env
   ```
3. Run the unified app:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
4. Open `http://localhost:8000` in your web browser.
