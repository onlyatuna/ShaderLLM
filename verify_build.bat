@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
echo [CHIMERA-V] Configuring...
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
echo [CHIMERA-V] Building...
cmake --build build --config Release
if %ERRORLEVEL% EQU 0 (
    echo [CHIMERA-V] Launching...
    .\build\bin\chimera_v_opt.exe
) else (
    echo [CHIMERA-V] Build Failed.
)
