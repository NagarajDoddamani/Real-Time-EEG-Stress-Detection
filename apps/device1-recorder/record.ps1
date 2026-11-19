param(
    [Parameter(Mandatory = $true)][int]$StreamIndex,
    [int]$DurationSeconds = 0,
    [string]$OutputDirectory = 'data/live/device1',
    [string]$ParticipantId,
    [string]$ParticipantName
)

if ($DurationSeconds -lt 0) {
    throw 'DurationSeconds must be 0 (manual stop) or positive.'
}
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
$cliArgs = @('-m', 'src.acquisition.cli', 'record', '--stream-index', "$StreamIndex",
    '--neo4j',
    '--output', $OutputDirectory)
if ($DurationSeconds -gt 0) { $cliArgs += @('--duration', "$DurationSeconds") }
if (-not $ParticipantId) { $ParticipantId = Read-Host 'Participant ID/code' }
if (-not $ParticipantId) { throw 'Participant ID/code is required for a Device 2 session.' }
$cliArgs += @('--participant-id', $ParticipantId)
if (-not $ParticipantName) { $ParticipantName = Read-Host 'User full name' }
if (-not $ParticipantName) { throw 'User full name is required so Device 2 can identify the session.' }
$cliArgs += @('--participant-name', $ParticipantName)
python @cliArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
