[CmdletBinding()]
param(
    [string]$PythonExe = $(
        if (Test-Path -Path ".\venv\Scripts\python.exe") {
            (Resolve-Path ".\venv\Scripts\python.exe").Path
        } else {
            "python"
        }
    ),
    [string]$TesseractPath = "C:\Program Files\Tesseract-OCR",
    [string]$LmStudioBaseUrl = "http://localhost:1234/v1",
    [string]$VlmModel = "qwen/qwen2.5-vl-7b-instruct",
    [string]$GenerationModel = "gemma-3",
    [switch]$DryRun,
    [switch]$SkipSwapPrompt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Invoke-Pytest {
    param([string[]]$Arguments)
    $command = @($PythonExe, "-m", "pytest") + $Arguments
    Write-Host ("Running: {0}" -f ($command -join " ")) -ForegroundColor DarkGray
    if (-not $DryRun) {
        & $PythonExe -m pytest @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Pytest failed with exit code $LASTEXITCODE"
        }
    }
}

if (Test-Path -Path $TesseractPath) {
    $env:Path = "$TesseractPath;$env:Path"
}

$env:VLM_ENABLED = "true"
$env:VLM_API_BASE = $LmStudioBaseUrl
$env:VLM_MODEL = $VlmModel
$env:LLM_PROVIDER = "lmstudio"
$env:LLM_BASE_URL = $LmStudioBaseUrl
$env:LLM_MODEL = $GenerationModel

Write-Step "Phase 1: OCR / VLM smoke test"
Write-Host "LM Studio model loaded: $VlmModel" -ForegroundColor Green
Write-Host "Run this phase with Qwen 2.5 VL loaded in LM Studio." -ForegroundColor Yellow
Invoke-Pytest -Arguments @("tests/test_ocr.py", "-q")

if (-not $SkipSwapPrompt) {
    Write-Step "Swap LM Studio model"
    Write-Host "Unload $VlmModel and load $GenerationModel in LM Studio, then press Enter to continue." -ForegroundColor Yellow
    [void](Read-Host "Press Enter after swapping the LM Studio model")
}

$env:VLM_ENABLED = "false"
$env:LLM_MODEL = $GenerationModel

Write-Step "Phase 2: Generation / API smoke test"
Write-Host "LM Studio model loaded: $GenerationModel" -ForegroundColor Green
Write-Host "Run this phase with Gemma loaded in LM Studio." -ForegroundColor Yellow
Invoke-Pytest -Arguments @("tests/test_api.py", "tests/test_retrieval.py", "tests/test_generation.py", "-q")

Write-Step "Done"
Write-Host "Both smoke-test phases completed successfully." -ForegroundColor Green
