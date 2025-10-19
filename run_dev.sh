
#!/usr/bin/env bash
set -e
echo "=== Backend ==="
cd backend
python -m venv .venv || true
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
cd ..

sleep 2
echo "=== Frontend ==="
cd frontend
if [ ! -d node_modules ]; then npm i; fi
npm run dev
