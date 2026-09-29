# Downloads the frozen camera detector `pipes run --detector on` loads:
# YOLOX-Nano, the ONNX file Megvii publishes as a release asset of YOLOX.
#
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_model.ps1
#
# Run it from the repository root. It writes models\yolox_nano.onnx and
# models\THIRD_PARTY.md; `models/` is gitignored, so the weights are never
# committed. The file is checked against a pinned sha256 and deleted if it
# does not match: a different file is a different model, and "frozen" would
# stop being true without anything saying so.
#
# Already present and matching: nothing is downloaded.

$ErrorActionPreference = 'Stop'

$Url    = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx'
$Sha256 = 'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d'
$Dir    = 'models'
$File   = Join-Path $Dir 'yolox_nano.onnx'

if (-not (Test-Path 'Cargo.toml')) {
    throw 'Run this from the repository root (the directory holding Cargo.toml).'
}
New-Item -ItemType Directory -Force $Dir | Out-Null

function Get-Sha256([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

if ((Test-Path $File) -and ((Get-Sha256 $File) -eq $Sha256)) {
    Write-Output "$File already present, sha256 $Sha256"
} else {
    $Part = "$File.part"
    Write-Output "downloading $Url"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Part
    $Got = Get-Sha256 $Part
    if ($Got -ne $Sha256) {
        Remove-Item -Force $Part
        throw "sha256 mismatch: expected $Sha256, got $Got. The file was deleted."
    }
    Move-Item -Force $Part $File
    Write-Output "$File written, sha256 $Sha256"
}

$Notice = @"
# Third-party files in models/

These files are downloaded by ``scripts/fetch_model.ps1`` and are never
committed.

## yolox_nano.onnx

- What: YOLOX-Nano, an 80-class COCO object detector, as the ONNX export
  Megvii publishes (opset 11, input ``images`` 1x3x416x416, output
  ``output`` 1x3549x85). ``pipes`` loads it at 1x3x192x640.
- From: $Url
  (release 0.1.1rc0 of https://github.com/Megvii-BaseDetection/YOLOX)
- sha256: $Sha256
- Size: 3,659,407 bytes.
- Trained on: COCO 2017.
- Licence of the code: Apache-2.0
  (https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE).
  The repository has no NOTICE file.
- Licence of the weights: **inferred, not stated.** The release carries no
  separate licence for the ONNX files; they are assets of the same
  repository, linked from its documentation as its pre-generated models, and
  are treated here as covered by the repository's Apache-2.0 grant. That is
  an inference from where they are published, not a quoted statement.
"@
Set-Content -Encoding utf8 -Path (Join-Path $Dir 'THIRD_PARTY.md') -Value $Notice
Write-Output "$(Join-Path $Dir 'THIRD_PARTY.md') written"
