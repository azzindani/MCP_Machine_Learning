@echo off
REM install.bat — MCP Machine Learning server installer for Windows
REM Requires: Python 3.12+, uv, git

setlocal EnableDelayedExpansion

set REPO_URL=https://github.com/azzindani/MCP_Machine_Learning.git
set INSTALL_DIR=%USERPROFILE%\.mcp_servers\MCP_Machine_Learning

echo === MCP Machine Learning Installer ===
echo.

REM --- Python version check ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://www.python.org/downloads/
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo ERROR: Python 3.12+ required. Found Python %PY_VER%.
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 12 (
    echo ERROR: Python 3.12+ required. Found Python %PY_VER%.
    exit /b 1
)
echo Python %PY_VER% OK

REM --- uv check ---
uv --version >nul 2>&1
if errorlevel 1 (
    echo uv not found. Installing uv...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
for /f "tokens=*" %%v in ('uv --version 2^>^&1') do echo uv %%v OK

REM --- VRAM detection ---
set VRAM_GB=0
set CONSTRAINED_MODE=0
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%m in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits 2^>nul') do set VRAM_MB=%%m
    if defined VRAM_MB (
        set /a VRAM_GB=!VRAM_MB! / 1024
        echo GPU VRAM: !VRAM_GB! GB
        if !VRAM_GB! LEQ 8 (
            set CONSTRAINED_MODE=1
            echo Constrained mode enabled ^(VRAM ^<^= 8 GB^)
        )
    )
)

REM --- Clone or update ---
if exist "%INSTALL_DIR%\.git" (
    echo Updating existing installation at %INSTALL_DIR%...
    cd /d "%INSTALL_DIR%"
    git fetch --quiet origin
    git reset --hard origin/main --quiet
) else (
    echo Cloning to %INSTALL_DIR%...
    git clone %REPO_URL% "%INSTALL_DIR%"
    cd /d "%INSTALL_DIR%"
)

REM --- Sync dependencies ---
echo Installing dependencies ^(uv sync^)...
uv sync --all-packages --quiet
echo Dependencies installed.

REM --- Write MCP config ---
echo.
echo === MCP Client Configuration ===
echo.
python install\mcp_config_writer.py --constrained %CONSTRAINED_MODE%

echo.
echo Installation complete!
echo.
echo Set MCP_CONSTRAINED_MODE=1 in your MCP client env block if you have ^<^= 8 GB VRAM.

endlocal
