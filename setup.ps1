# Requires -RunAsAdministrator
# Self-elevate if not running as Admin
$currUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currUser)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
  Write-Host "Requesting admin privileges..."
  Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
  exit
}

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSCommandPath
Set-Location $root

function Have-Cmd($name) {
  return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

# ---------- 1) Ensure Node.js & npm ----------
if (Have-Cmd "node" -and Have-Cmd "npm") {
  Write-Host "Node.js and npm already installed."
} else {
  Write-Host "Node.js/npm not found. Downloading Node LTS..."
  $nodeVersion = "v20.16.0"
  $msiUrl = "https://nodejs.org/dist/$nodeVersion/node-$nodeVersion-x64.msi"
  $tmp = Join-Path $env:TEMP "node-$nodeVersion-x64.msi"
  Invoke-WebRequest -Uri $msiUrl -OutFile $tmp
  Write-Host "Installing Node.js silently..."
  Start-Process msiexec.exe -Wait -ArgumentList "/i `"$tmp`" /quiet /norestart"
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  if (-not (Have-Cmd "node")) { throw "Node installation failed: 'node' not found in PATH." }
  if (-not (Have-Cmd "npm"))  { throw "Node installation failed: 'npm' not found in PATH." }
  Write-Host "Node.js/npm installed."
}

# ---------- 2) Backend (Python) ----------
# Pick python executable
$python = $null
if (Have-Cmd "py") {
  try {
    & py -3.11 -c "import sys;print(sys.version)" *> $null
    if ($LASTEXITCODE -eq 0) { $python = "py -3.11" }
  } catch {}
  if (-not $python) { $python = "py -3" }
} elseif (Have-Cmd "python") {
  $python = "python"
} elseif (Have-Cmd "python3") {
  $python = "python3"
} else {
  throw "Python not found. Please install Python 3.11 and run again."
}

$backend = Join-Path $root "backend"
if (-not (Test-Path $backend)) { throw "Folder 'backend' not found at repo root." }

Write-Host "`n=== Backend: create venv and install requirements ==="
Set-Location $backend

# Create venv if missing
if (-not (Test-Path ".venv")) {
  iex "$python -m venv .venv"
}
# Activate venv for this session
$venvActivate = Join-Path $backend ".venv\Scripts\Activate.ps1"
. $venvActivate

# Upgrade pip and install requirements (if present)
python -m pip install --upgrade pip
if (Test-Path "requirements.txt") {
  pip install -r requirements.txt
} else {
  Write-Host "No requirements.txt in backend - skip."
}

# ---------- 3) npm install ----------
function Run-NpmInstallIfPackageJson($dir) {
  if (Test-Path (Join-Path $dir "package.json")) {
    Write-Host "`n=== npm install in $dir ==="
    Push-Location $dir
    npm install
    Pop-Location
  } else {
    Write-Host "No package.json in $dir - skip npm install."
  }
}

# frontend
$frontend = Join-Path $root "frontend"
if (Test-Path $frontend) { Run-NpmInstallIfPackageJson $frontend }

# backend (only if backend has package.json)
Run-NpmInstallIfPackageJson $backend

# ---------- 4) Done ----------
Set-Location $root
Write-Host "`nAll set! "
Write-Host "Backend:"
Write-Host "  .\backend\.venv\Scripts\Activate.ps1"
Write-Host "  uvicorn app.main:app --app-dir backend --reload"
Write-Host "Frontend:"
Write-Host "  cd frontend"
Write-Host "  npm run dev"
