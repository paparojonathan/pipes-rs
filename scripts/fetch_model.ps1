# Downloads the camera detectors `pipes run --detector-model` loads: the ONNX
# files Megvii publishes for YOLOX-Nano, -S, -M, -L and -X (754 MB in all, four
# fifths of it the two biggest, which only a GPU run loads), then derives the
# files `pipes` loads from them with scripts/prepare_models.py. That script
# also downloads and derives the lidar detector (`pipes run --lidar-detector
# on`, models\pointpillars-kitti.onnx and its 16-bit twin for the GPU), so one
# run of this fetches everything.
#
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_model.ps1
#
# Run it from the repository root. It writes models\yolox_nano.onnx and the
# other four published files, each checked against a pinned sha256 and deleted
# if it does not match -- a different file is a different model, and "frozen"
# would stop being true without anything saying so -- then runs
# `python scripts\prepare_models.py`, which needs the `onnx` package and, for
# the GPU's files of 16-bit weights, `onnxconverter-common`:
#
#   pip install onnx==1.17.0 onnxconverter-common==1.14.0 numpy==2.1.3
#
# It writes the derived files (yolox_nano-640x192.onnx and the others) beside
# them. `models/` is gitignored, so no weights are ever committed. Already
# present and matching: nothing is downloaded.
#
# Docker users need none of this: the image downloads and derives the models
# when it is built.

$ErrorActionPreference = 'Stop'

$Release = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/'
$Sources = @{
    'yolox_nano.onnx' = 'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d'
    'yolox_s.onnx'    = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
    'yolox_m.onnx'    = '21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f'
    'yolox_l.onnx'    = '7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411'
    'yolox_x.onnx'    = 'c892d7aaf1c4746d8a4d675bec669a4db4f434b4ee1efb654bc9b353379c7c55'
}
$Dir = 'models'

if (-not (Test-Path 'Cargo.toml')) {
    throw 'Run this from the repository root (the directory holding Cargo.toml).'
}
New-Item -ItemType Directory -Force $Dir | Out-Null

function Get-Sha256([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
foreach ($Name in $Sources.Keys) {
    $File = Join-Path $Dir $Name
    $Sha256 = $Sources[$Name]
    if ((Test-Path $File) -and ((Get-Sha256 $File) -eq $Sha256)) {
        Write-Output "$File already present, sha256 $Sha256"
        continue
    }
    $Part = "$File.part"
    Write-Output "downloading $Release$Name"
    Invoke-WebRequest -UseBasicParsing -Uri "$Release$Name" -OutFile $Part
    $Got = Get-Sha256 $Part
    if ($Got -ne $Sha256) {
        Remove-Item -Force $Part
        throw "sha256 mismatch for ${Name}: expected $Sha256, got $Got. The file was deleted."
    }
    Move-Item -Force $Part $File
    Write-Output "$File written, sha256 $Sha256"
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $Python) {
    Write-Output 'python not found: install Python 3 with `pip install onnx==1.17.0 onnxconverter-common==1.14.0 numpy==2.1.3`, then run `python scripts\prepare_models.py` to derive the files pipes loads.'
    exit 0
}
& python scripts\prepare_models.py
if ($LASTEXITCODE -ne 0) {
    throw 'scripts\prepare_models.py failed (are its packages installed? pip install onnx==1.17.0 onnxconverter-common==1.14.0 numpy==2.1.3)'
}
