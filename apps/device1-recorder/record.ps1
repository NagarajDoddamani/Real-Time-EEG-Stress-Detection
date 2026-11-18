param(
    [Parameter(Mandatory = $true)][int]$StreamIndex,
    [int]$DurationSeconds = 60,
    [int]$StoreRawSeconds = 60,
    [string]$OutputDirectory = 'data/live/device1',
    [string]$ParticipantId,
    [string]$ParticipantName
)

if ($DurationSeconds -lt 1 -or $StoreRawSeconds -lt 0) {
    throw 'DurationSeconds must be positive and StoreRawSeconds cannot be negative.'
}
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
$cliArgs = @('-m', 'src.acquisition.cli', 'record', '--stream-index', "$StreamIndex",
    '--duration', "$DurationSeconds", '--neo4j', '--db-raw-seconds', "$StoreRawSeconds",
    '--output', $OutputDirectory)
if (-not $ParticipantId) { $ParticipantId = Read-Host 'Participant ID/code' }
if (-not $ParticipantId) { throw 'Participant ID/code is required for a Device 2 session.' }
$cliArgs += @('--participant-id', $ParticipantId)
if (-not $ParticipantName) { $ParticipantName = Read-Host 'Participant display name (optional; leave blank to omit)' }
if ($ParticipantName) { $cliArgs += @('--participant-name', $ParticipantName) }
python @cliArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
