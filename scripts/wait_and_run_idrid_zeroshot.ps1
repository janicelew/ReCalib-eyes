# Wait for eyeclip_visual.pt then run IDRiD zero-shot pipeline.
$ErrorActionPreference = "Stop"
$Checkpoint = "D:\Projects\EyeCLIP\eyeclip_visual.pt"
$Repo = "D:\Projects\ReCalib-eyes"
$MinBytes = 2000000000

Write-Host "Waiting for checkpoint at $Checkpoint ..."
Write-Host "Download in browser: https://drive.google.com/file/d/1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr/view"

while ($true) {
    if ((Test-Path $Checkpoint) -and (Get-Item $Checkpoint).Length -ge $MinBytes) {
        Write-Host "Checkpoint found: $((Get-Item $Checkpoint).Length) bytes"
        break
    }
    Start-Sleep -Seconds 10
}

Set-Location $Repo
powershell -ExecutionPolicy Bypass -File scripts\run_idrid_zeroshot_pipeline.ps1
