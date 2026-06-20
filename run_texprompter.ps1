# Launcher for TexPrompter on Windows PowerShell.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$Root;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $Root
}
$env:PIPELINE_API_ENABLED = '1'

Write-Host "Starting TexPrompter backend API..."
$apiProcess = Start-Process -FilePath python -ArgumentList '-m', 'uvicorn', 'services.api:app', '--host', '127.0.0.1', '--port', '8000' -NoNewWindow -PassThru

$healthUrl = 'http://127.0.0.1:8000/health'
$maxAttempts = 20
$attempt = 0

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Milliseconds 500
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            break
        }
    } catch {
        # continue waiting
    }
    $attempt++
}

if ($attempt -ge $maxAttempts) {
    Write-Error "FastAPI backend did not start in time."
    if ($apiProcess -and -not $apiProcess.HasExited) {
        $apiProcess.Kill()
    }
    exit 1
}

Write-Host "FastAPI backend ready at http://127.0.0.1:8000"
Write-Host "Launching Streamlit UI..."
streamlit run app/streamlit_app.py

if ($apiProcess -and -not $apiProcess.HasExited) {
    $apiProcess.Kill()
}
