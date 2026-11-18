param(
    [Parameter(Mandatory = $true)][int]$StreamIndex,
    [int]$DurationSeconds = 0,
    [int]$StoreRawSeconds = 1200,
    [string]$OutputDirectory = 'data/live/device1',
    [string]$ParticipantId,
    [string]$ParticipantName
)

if ($DurationSeconds -lt 0 -or $StoreRawSeconds -lt 1) {
    throw 'DurationSeconds must be 0 (manual stop) or positive; StoreRawSeconds must be positive.'
}
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
$cliArgs = @('-m', 'src.acquisition.cli', 'record', '--stream-index', "$StreamIndex",
    '--neo4j', '--db-raw-seconds', "$StoreRawSeconds",
    '--output', $OutputDirectory)
if ($DurationSeconds -gt 0) { $cliArgs += @('--duration', "$DurationSeconds") }
if (-not $ParticipantId) { $ParticipantId = Read-Host 'Participant ID/code' }
if (-not $ParticipantId) { throw 'Participant ID/code is required for a Device 2 session.' }
$cliArgs += @('--participant-id', $ParticipantId)
if (-not $ParticipantName) { $ParticipantName = Read-Host 'Participant display name (optional; leave blank to omit)' }
if ($ParticipantName) { $cliArgs += @('--participant-name', $ParticipantName) }
python @cliArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
