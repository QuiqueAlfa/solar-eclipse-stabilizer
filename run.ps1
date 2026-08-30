# Optional Windows helper for solar-eclipse-stabilizer.
# Sets up the virtualenv/dependencies and forwards an explicit command + VIDEO.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ValueOptions = @(
    "--profile", "--out", "--analysis-width", "--radius", "--min-quality",
    "--samples", "--start-frame", "--end-frame",
    "--debug-width", "--debug-max-images",
    "--drop-frames", "--keep-frames",
    "--preview-width", "--speed",
    "--name", "--crf", "--preset", "--threads"
)

function Get-PositionalArguments {
    $positional = [System.Collections.Generic.List[string]]::new()
    $i = 0
    while ($i -lt $args.Count) {
        $token = [string]$args[$i]
        if ($token.StartsWith("-") -and $token -ne "-") {
            if ($ValueOptions -contains $token -and $i + 1 -lt $args.Count) { $i++ }
        } else {
            $positional.Add($token)
        }
        $i++
    }
    return $positional
}

$Usage = @"
Usage:
  .\run.ps1 [global options] COMMAND VIDEO [subcommand options]

Recommended workflow (preview -> visual review -> export):
  preview   analyze if needed, then build a fast low-resolution preview
  export    export at original resolution after approving the preview

Advanced diagnostics:
  inspect   inspect isolated frames without decoding the whole sequence
  analyze   track the whole video and optionally write detailed debug artifacts

Global options (before COMMAND):
  --profile FILE --out DIR --analysis-width N --radius R
  --min-quality Q --force --no-auto-repair

Example:
  .\run.ps1 preview VIDEO
  # Watch and approve preview.mp4
  .\run.ps1 export VIDEO

Advanced examples:
  .\run.ps1 inspect VIDEO
  .\run.ps1 analyze VIDEO --debug --debug-width 320
  .\run.ps1 preview VIDEO --drop-frames 20-24,700 --keep-frames 220
  .\run.ps1 export VIDEO --drop-frames 20-24,700 --keep-frames 220

A command and an explicit VIDEO are required; no video is ever auto-detected.
"@

$Positional = @(Get-PositionalArguments @args)
if ($Positional.Count -lt 2) {
    Write-Host $Usage
    exit 1
}
$Command = $Positional[0]
$Video = $Positional[1]
if ($Command -notin @("inspect", "analyze", "preview", "export")) {
    Write-Host "Unknown command: $Command"
    Write-Host $Usage
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    python -m venv .venv
}

Write-Host "Installing dependencies into .venv ..."
& ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running eclipse_stabilizer.py $args ..."
& ".venv\Scripts\python.exe" eclipse_stabilizer.py @args
exit $LASTEXITCODE
