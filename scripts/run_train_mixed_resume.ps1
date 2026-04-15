#Requires -Version 5.1
<#
.SYNOPSIS
  Resume CNR+PKLot mixed YOLO training (train_mixed_final) in foreground.

.USAGE
  cd E:\Programs\download\ps2.0\parking-slot-yolov10
  .\scripts\run_train_mixed_resume.ps1

  Or nested (works even if $PSScriptRoot is empty in child pwsh):
  powershell -ExecutionPolicy Bypass -File .\scripts\run_train_mixed_resume.ps1

  Custom Python:
  .\scripts\run_train_mixed_resume.ps1 -PythonExe "E:\miniconda3\envs\parking_env\python.exe"
#>
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

# Resolve repo root: prefer $PSScriptRoot; fallback to this script's directory (nested -File safe)
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    if ($PSCommandPath) {
        # PS 5.1: do not combine -LiteralPath with -Parent (ambiguous parameter set)
        $ScriptDir = Split-Path -Parent -Path $PSCommandPath
    } elseif ($MyInvocation.MyCommand.Path) {
        $ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
    }
}
if (-not $ScriptDir) {
    Write-Error "Cannot resolve script directory. Run from repo: .\scripts\run_train_mixed_resume.ps1"
}

$RepoRoot = Split-Path -Parent -Path $ScriptDir
Set-Location -LiteralPath $RepoRoot

if (-not $PythonExe) {
    $candidates = @(
        "E:\miniconda3\envs\parking_env\python.exe",
        "$env:USERPROFILE\miniconda3\envs\parking_env\python.exe",
        "$env:USERPROFILE\anaconda3\envs\parking_env\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { $PythonExe = $c; break }
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python not found. Pass -PythonExe path to parking_env python.exe"
}

$lastPt = Join-Path $RepoRoot "runs\detect\train_mixed_final\weights\last.pt"
if (-not (Test-Path -LiteralPath $lastPt)) {
    Write-Error "Checkpoint not found: $lastPt"
}

# Forward slashes for Python raw string (avoids escape issues)
$lastPtForPy = ($lastPt | Resolve-Path).Path -replace '\\', '/'

Write-Host "Repo:   $RepoRoot"
Write-Host "Python: $PythonExe"
Write-Host "Resume: $lastPtForPy"
Write-Host "Starting resume (do not start a second trainer for the same run).`n"

$pyCode = "from ultralytics import YOLO; YOLO(r'$lastPtForPy').train(resume=True, device=0)"
& $PythonExe -c $pyCode
