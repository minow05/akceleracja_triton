# One command: pull -> collect all report data -> commit -> push.
# Run from an *Administrator* PowerShell (needed for Nsight Compute GPU counters):
#     powershell -ExecutionPolicy Bypass -File .\run_report.ps1
# Extra arguments are passed to report_bench.py, e.g.  -File .\run_report.ps1 --no-ncu
Set-Location $PSScriptRoot

git pull
if ($LASTEXITCODE -ne 0) { Write-Host "git pull failed" -ForegroundColor Red; exit 1 }

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py report_bench.py @args
if ($LASTEXITCODE -ne 0) {
    Write-Host "report_bench.py failed - see report_data\log.txt (results NOT pushed)" -ForegroundColor Red
    exit 1
}

git add -f report_data
git commit -m "Add report benchmark data (report_bench.py)"
git push
if ($LASTEXITCODE -ne 0) { Write-Host "git push failed - push manually" -ForegroundColor Red; exit 1 }
Write-Host "Done - report_data pushed." -ForegroundColor Green
