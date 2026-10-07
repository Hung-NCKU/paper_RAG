$ErrorActionPreference = 'Stop'
# WSL 會把反斜線當成跳脫字元，先換成正斜線（D:/claude/...）再交給 wslpath
$windowsProject = $PSScriptRoot -replace '\\', '/'
$linuxProject = & wsl -d Ubuntu-24.04 -- wslpath -a $windowsProject
if ($LASTEXITCODE -ne 0 -or -not $linuxProject) { throw 'Cannot locate project in WSL.' }
$linuxProject = $linuxProject.Trim()
$linuxAddress = ((& wsl -d Ubuntu-24.04 -- hostname -I).Trim() -split '\s+')[0]
if ($LASTEXITCODE -ne 0 -or -not $linuxAddress) { throw 'Cannot read WSL address.' }
Write-Host "Paper RAG: http://${linuxAddress}:8001"
& wsl -d Ubuntu-24.04 -- env "BIND_HOST=$linuxAddress" PORT=8001 bash "$linuxProject/start-wsl.sh"
