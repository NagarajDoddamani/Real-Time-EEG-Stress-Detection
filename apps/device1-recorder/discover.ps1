Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
python -m src.acquisition.cli discover
