Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
python -m src.acquisition.stream_monitor_ui
