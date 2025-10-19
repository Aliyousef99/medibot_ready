
# Windows PowerShell dev helper
Write-Host "=== Backend ==="
Push-Location backend
if (-not (Test-Path .venv)) { python -m venv .venv }
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
Start-Process powershell -ArgumentList '-NoExit','-Command','cd backend; . .venv/Scripts/Activate.ps1; uvicorn app:app --host 0.0.0.0 --port 8000 --reload'
Pop-Location

Start-Sleep -Seconds 2

Write-Host "=== Frontend ==="
Push-Location frontend
if (-not (Test-Path node_modules)) { npm i }
npm run dev
Pop-Location
