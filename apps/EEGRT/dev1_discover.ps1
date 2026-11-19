$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = 'python' }
& $pythonExe (Join-Path $PSScriptRoot 'eeg_cli.py') discover
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
