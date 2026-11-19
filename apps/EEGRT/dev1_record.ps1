param(
    [Parameter(Mandatory = $true)][int]$StreamIndex,
    [string]$ParticipantId,
    [string]$ParticipantName,
    [int]$DurationSeconds = 0
)

$ErrorActionPreference = 'Stop'
if ($DurationSeconds -lt 0) { throw 'DurationSeconds must be 0 (manual stop) or positive.' }
Set-Location $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = 'python' }
if (-not $ParticipantId) { $ParticipantId = Read-Host 'Participant ID/code' }
if (-not $ParticipantName) { $ParticipantName = Read-Host 'Participant full name' }
if (-not $ParticipantId -or -not $ParticipantName) { throw 'Participant ID and full name are required.' }

$cliArgs = @((Join-Path $PSScriptRoot 'eeg_cli.py'), 'record', '--stream-index', "$StreamIndex",
    '--participant-id', $ParticipantId, '--participant-name', $ParticipantName,
    '--neo4j', '--output', (Join-Path $PSScriptRoot 'data\live\dev1-recordings'))
if ($DurationSeconds -gt 0) { $cliArgs += @('--duration', "$DurationSeconds") }
& $pythonExe @cliArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
