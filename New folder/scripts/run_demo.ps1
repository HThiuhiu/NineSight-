# run_demo.ps1 - chay toan bo pipeline: build -> test -> benchmark -> EVT -> bieu do.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 [-LongSeconds 300] [-StallMode global]
#
# Ba lan do open-loop:
#   dense  : rate cao, thoi gian ngan  (10k mau lien tiep, chi chua vai stall)
#   sparse : CUNG rate, thoi gian dai hon, chi ghi 1/K request (10k mau trai deu, ~40 stall doc lap)
#            -> cung tai voi ground truth (rate thap hon se doi phan phoi vi backlog sau moi pause)
#   long   : rate cao, thoi gian dai   (ground truth)
param(
    [int]$Rate = 2000,
    [int]$DenseSeconds = 5,         # 5s * 2000 rps  = 10k mau
    [int]$SparseSeconds = 100,      # 100s * 2000 rps = 200k request, ghi 1/20 = 10k mau
    [int]$SparseEvery = 20,
    [int]$LongSeconds = 300,        # 300s * 2000 rps = 600k mau
    [int]$ClosedRequests = 30000,
    [string]$StallMode = "global",  # "request" = dung spec 1%/request (capacity ~300 rps/conn)
    [int]$StallMinMs = 100,
    [int]$StallMaxMs = 500,
    [int]$GcIntervalMs = 2000
)
# Khong dung ErrorActionPreference=Stop: PowerShell 5.1 coi moi dong stderr cua native exe
# (vd unittest) la loi. Kiem tra $LASTEXITCODE tuong minh thay vao do.
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Step "build + unit tests"
mingw32-make all test 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) { throw "build/test failed" }

$srvArgs = "--port 8080 --stall-mode $StallMode --stall-min-ms $StallMinMs --stall-max-ms $StallMaxMs --gc-interval-ms $GcIntervalMs"
$server = Start-Process -FilePath build\dummy_server.exe -ArgumentList $srvArgs -PassThru -NoNewWindow
Start-Sleep -Milliseconds 500
try {
    Step "Task 2: closed-loop"
    build\closed_loop.exe --requests $ClosedRequests --out results/closed

    Step "Task 3: open-loop DENSE sample ($Rate rps x $DenseSeconds s)"
    build\open_loop.exe --rate $Rate --duration-s $DenseSeconds --warmup-s 1 --out results/open_dense --raw-out results/open_dense_raw.csv

    Step "Task 3: open-loop SPARSE sample ($Rate rps x $SparseSeconds s, raw 1/$SparseEvery)"
    build\open_loop.exe --rate $Rate --duration-s $SparseSeconds --warmup-s 1 --out results/open_sparse --raw-out results/open_sparse_raw.csv --raw-every $SparseEvery

    Step "Task 5: open-loop LONG run / ground truth ($Rate rps x $LongSeconds s)"
    $rawMax = $Rate * $LongSeconds + 10000
    build\open_loop.exe --rate $Rate --duration-s $LongSeconds --warmup-s 1 --out results/open_long --raw-out results/open_long_raw.csv --raw-max $rawMax
}
finally {
    Stop-Process -Id $server.Id -Force
}

Step "Task 4: EVT on sparse sample"
python analysis\evt_gpd.py results\open_sparse_raw.csv --sample-size 10000

Step "Task 5: validation plots"
$theory = @()
if ($StallMode -eq "global") { $theory = @("--theory", "$StallMinMs,$StallMaxMs,$GcIntervalMs") }
python analysis\plot_validation.py --closed results\closed_buckets.csv --open results\open_long_buckets.csv `
    --dense-raw results\open_dense_raw.csv --sparse-raw results\open_sparse_raw.csv `
    --long-raw results\open_long_raw.csv --outdir results @theory
