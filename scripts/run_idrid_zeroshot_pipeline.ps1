# Run full IDRiD zero-shot pipeline (Windows)
$ErrorActionPreference = "Stop"
$Repo = "D:\Projects\ReCalib-eyes"
$RawBase = "https://raw.githubusercontent.com/janicelew/ReCalib-eyes/main"

Set-Location $Repo
$env:PYTHONPATH = "src"

Write-Host "=== Step 0: download repo assets ===" -ForegroundColor Cyan
$downloads = @(
  "results/aptos2019_zeroshot/predictions_APTOS2019.csv",
  "results/aptos2019_zeroshot/metrics.csv",
  "scripts/prepare_idrid.py",
  "scripts/download_idrid.py"
)
foreach ($rel in $downloads) {
  $dest = Join-Path $Repo $rel
  New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
  if (-not (Test-Path $dest) -or ((Get-Item $dest).Length -lt 100)) {
    curl.exe -fsSL --retry 5 -o $dest "$RawBase/$rel"
  }
}

Write-Host "=== Step 1: validate IDRiD ===" -ForegroundColor Cyan
py -3.13 scripts/validate_idrid.py

Write-Host "=== Step 2: APTOS source calibration ===" -ForegroundColor Cyan
py -3.13 scripts/calibrate_source_predictions.py `
  --predictions results/aptos2019_zeroshot/predictions_APTOS2019.csv `
  --source-name APTOS2019 `
  --output-dir outputs/aptos2019_source_calibration

Write-Host "=== Step 3: IDRiD zero-shot ===" -ForegroundColor Cyan
py -3.13 -m recalib_eye.zeroshot_dr --config configs/idrid_zeroshot.json

Write-Host "=== Step 4: apply source calibration to IDRiD ===" -ForegroundColor Cyan
py -3.13 scripts/apply_source_calibration.py `
  --predictions outputs/idrid_zeroshot/predictions_IDRiD.csv `
  --calibration outputs/aptos2019_source_calibration/source_calibration.json `
  --output-dir outputs/idrid_zeroshot_calibrated `
  --dataset-name IDRiD

Write-Host "=== Step 5: package deliverables ===" -ForegroundColor Cyan
$out = Join-Path $Repo "results/idrid_zeroshot"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Copy-Item outputs/idrid_zeroshot/predictions_IDRiD.csv $out -Force
Copy-Item outputs/idrid_zeroshot/metrics.csv $out -Force
Copy-Item outputs/idrid_zeroshot_calibrated/metrics_calibrated.csv $out -Force
Copy-Item configs/idrid_zeroshot.json (Join-Path $out "idrid_zeroshot.json") -Force
Copy-Item outputs/aptos2019_source_calibration/source_calibration.json $out -Force

Write-Host "Done. Deliverables in $out"
