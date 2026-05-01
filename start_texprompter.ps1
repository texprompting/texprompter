# Dev launcher (Windows): SSH tunnel + MLflow + interactive menu. See README «Dev startup scripts».
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$Root;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = $Root
}
Write-Host "[dev] Tip: conda activate texprompting if python/mlflow imports fail." -ForegroundColor DarkGray
python (Join-Path $Root "scripts\texprompter_dev.py") @args
