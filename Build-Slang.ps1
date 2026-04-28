# Build-Slang.ps1 - CHIMERA-V Production Grade Compiler Pipeline
# 廢除源碼目錄污染，實施執行環境同步策略。

param (
    [string]$SourceDir = "src\shaders",
    [string]$BuildDir = "build" # 預設 CMake 建置目錄
)

# --- 🚀 工程化路徑自動偵測 ---
$PossibleBinDirs = @(
    "$BuildDir\bin",
    "out\build\x64-Release\bin",
    "bin"
)

$BinDir = $null
foreach ($dir in $PossibleBinDirs) {
    if (Test-Path $dir) {
        $BinDir = $dir
        break
    }
}

if (-not $BinDir) {
    Write-Host "!!! WARNING: Build output directory not found. Defaulting to 'bin'..." -ForegroundColor Yellow
    $BinDir = "bin"
}

$OutputDir = Join-Path $BinDir "src\shaders"
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

# 核心路徑對齊
$SlangPath = "raw\slang-2026.7-windows-x86_64\bin\slangc.exe"
$SlangInclude = "raw\slang-2026.7-windows-x86_64\bin\slang-standard-module-2026.7"
$GlslangPath = "glslangValidator"
if ($env:VULKAN_SDK) { $GlslangPath = Join-Path $env:VULKAN_SDK "Bin\glslangValidator.exe" }

Write-Host "========== CHIMERA-V: Secure Compiler Pipeline ==========" -ForegroundColor Cyan
Write-Host "Target Environment: $BinDir" -ForegroundColor Gray

# 清理舊有的 .spv 確保不留殘留
Get-ChildItem -Path $OutputDir -Filter "*.spv" | Remove-Item -Force

$ShaderFiles = Get-ChildItem -Path $SourceDir -Filter "*.comp"
$ShaderFiles += Get-ChildItem -Path $SourceDir -Filter "*.slang" 

foreach ($file in $ShaderFiles) {
    $outFile = Join-Path -Path $OutputDir -ChildPath "$($file.BaseName).spv"
    Write-Host "Compiling [$($file.Name)] -> [bin/src/shaders/...] " -NoNewline

    if ($file.Extension -eq ".comp") {
        if ($file.BaseName -match "^(nca_evolve|coopmat_test)$") {
            & $GlslangPath -V --target-env vulkan1.3 $file.FullName -o $outFile 2>&1 | Out-Null
        } else {
            & $SlangPath $file.FullName -allow-glsl -I $SlangInclude -target spirv -profile glsl_460 -O3 -o $outFile 2>&1 | Out-Null
        }
    } else {
        & $SlangPath $file.FullName -I $SlangInclude -target spirv -profile spirv_1_5+cooperative_matrix -O3 -o $outFile 2>&1 | Out-Null
    }

    if ($LASTEXITCODE -eq 0 -and (Test-Path $outFile)) {
        $size = (Get-Item $outFile).Length
        Write-Host "OK ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "FAILED" -ForegroundColor Red
        exit 1 
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan

