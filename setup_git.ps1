[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

if ($null -eq (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found on PATH. Install Git for Windows, restart PowerShell, and rerun this script."
}

Write-Host "Initializing Git repository in $PSScriptRoot" -ForegroundColor Cyan
git init
if ($LASTEXITCODE -ne 0) { throw "git init failed with exit code $LASTEXITCODE." }

Write-Host "Staging project files (honoring .gitignore)" -ForegroundColor Cyan
git add .
if ($LASTEXITCODE -ne 0) { throw "git add failed with exit code $LASTEXITCODE." }

$stagedFiles = @(git diff --cached --name-only)
if ($stagedFiles.Count -eq 0) {
    throw "No files are staged. Check .gitignore and the project directory before committing."
}

Write-Host "Creating initial commit" -ForegroundColor Cyan
git commit -m "feat: Initial commit of Distributed Semantic Inference Router"
if ($LASTEXITCODE -ne 0) {
    throw "git commit failed. Configure git user.name and user.email, then rerun setup_git.ps1."
}

Write-Host "`nInitial commit created successfully. Create an empty public repository on GitHub.com, then run these exact commands:`n" -ForegroundColor Green
Write-Host 'git remote add origin https://github.com/<GITHUB_USERNAME>/<REPOSITORY_NAME>.git' -ForegroundColor Yellow
Write-Host 'git branch -M main' -ForegroundColor Yellow
Write-Host 'git push -u origin main' -ForegroundColor Yellow
