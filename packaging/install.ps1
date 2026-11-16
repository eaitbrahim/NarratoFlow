# NarratoFlow installer for Windows: installs every extra, ffmpeg with libass, the C++ build
# tools and the CAMeL dialect model. Run from the unpacked bundle:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
# Safe to run again: each step is skipped when it is already done.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$UvVersion = '0.11.31'
$CamelModel = 'dialectid-model26'
$Here = $PSScriptRoot

function Say([string]$Message) { Write-Host "`n==> $Message" }
function Fail([string]$Message) { Write-Host "`nERROR: $Message" -ForegroundColor Red; exit 1 }
function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$File $($Arguments -join ' ') exited with $LASTEXITCODE" }
}
function Add-SessionPath([string]$Dir) {
    if ((Test-Path $Dir) -and -not (($env:Path -split ';') -contains $Dir)) { $env:Path = "$Dir;$env:Path" }
}
function Test-Libass([string]$Ffmpeg) {
    # Windows PowerShell 5.1 turns redirected stderr into terminating errors under 'Stop'.
    $ErrorActionPreference = 'Continue'
    $filters = & $Ffmpeg -hide_banner -filters 2>$null
    return [bool]($filters | Where-Object { ($_ -split '\s+')[2] -eq 'ass' })
}
function Find-Ffmpeg {
    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($cmd -and (Test-Libass $cmd.Source)) { return $cmd.Source }
    return $null
}

$wheels = @(Get-ChildItem -Path $Here -Filter 'narratoflow-*-py3-none-any.whl')
if ($wheels.Count -ne 1) { Fail "expected exactly one narratoflow wheel in $Here" }
$Wheel = $wheels[0].FullName
$Constraints = Join-Path $Here 'constraints.txt'
if (-not (Test-Path $Constraints)) { Fail "constraints.txt is missing from $Here" }
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { Fail 'a 64-bit x86 (AMD64) Windows is required' }

$winget = Get-Command winget -ErrorAction SilentlyContinue
$wingetArgs = @('--exact', '--silent', '--accept-source-agreements', '--accept-package-agreements')

# The align extra (ctc-forced-aligner) is built from source and needs the MSVC C++ tools.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$hasMsvc = (Test-Path $vswhere) -and
    (& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if (-not $hasMsvc) {
    if (-not $winget) { Fail 'install "Visual Studio Build Tools" with the "Desktop development with C++" workload, then run this installer again' }
    Say 'Installing the Visual Studio C++ Build Tools (approve the administrator prompt; this takes a while)'
    Invoke-Checked 'winget' (@('install', '--id', 'Microsoft.VisualStudio.2022.BuildTools') + $wingetArgs + @(
        '--override', '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'))
}

$Ffmpeg = Find-Ffmpeg
if (-not $Ffmpeg) {
    Say 'Installing ffmpeg with libass'
    if ($winget) {
        Invoke-Checked 'winget' (@('install', '--id', 'Gyan.FFmpeg') + $wingetArgs)
        Add-SessionPath (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links')
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Invoke-Checked 'choco' @('install', 'ffmpeg-full', '-y', '--no-progress')
        Add-SessionPath (Join-Path $env:ProgramData 'chocolatey\bin')
    } else {
        Fail 'install ffmpeg with libass (for example the gyan.dev "full" build) and put it on PATH, then run this installer again'
    }
    $Ffmpeg = Find-Ffmpeg
    if (-not $Ffmpeg) { Fail 'ffmpeg with libass is still not on PATH; open a new PowerShell window and run this installer again' }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv $UvVersion"
    Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1" | Invoke-Expression
}
Add-SessionPath (Join-Path $env:USERPROFILE '.local\bin')

Say 'Installing NarratoFlow with every extra (several GB on first install)'
Invoke-Checked 'uv' @('tool', 'install', '--force', '--python', '3.12', '--constraints', $Constraints, "narratoflow[all] @ $Wheel")
try { & uv tool update-shell *> $null } catch { Write-Host 'Run "uv tool update-shell" if narratoflow is not found in a new window.' }

$EnvScripts = Join-Path (& uv tool dir) 'narratoflow\Scripts'
Say "Downloading the CAMeL dialect model ($CamelModel, 274 MB)"
Invoke-Checked (Join-Path $EnvScripts 'camel_data.exe') @('-i', $CamelModel)

Say 'Checking the installation'
Invoke-Checked (Join-Path $EnvScripts 'python.exe') @('-c', 'import camel_tools, ctc_forced_aligner, diffusers, edge_tts, gradio, mishkal, piper, torch, transformers')
Invoke-Checked (Join-Path (& uv tool dir --bin) 'narratoflow.exe') @('--version')

Say 'Done. Open a new PowerShell window, then run: narratoflow --help'
