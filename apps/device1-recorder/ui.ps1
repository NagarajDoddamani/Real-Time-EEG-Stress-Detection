$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location (Join-Path $projectRoot 'apps\EEGRT')
npm start
