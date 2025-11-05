@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PWSH_EXE="
set "NEEDS_VENV="

if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" set "NEEDS_VENV=1"

for /f "delims=" %%I in ('where pwsh 2^>nul') do (
    set "PWSH_EXE=%%I"
    goto :checkvenv
)

echo.
echo ERROR: PowerShell 7 (pwsh) is required but was not found on PATH.
echo        Install it from https://aka.ms/powershell then re-run RunBot.bat.
goto :pauseexit

:checkvenv
if defined NEEDS_VENV (
    call :create_venv
    if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" goto :pauseexit
)
goto :run

:create_venv
echo.
echo Creating virtual environment at %SCRIPT_DIR%.venv ...
set "PY_CMD="
for /f "delims=" %%P in ('where py 2^>nul') do (
    set "PY_CMD=%%P"
    goto :use_py_launcher
)
goto :check_python_fallback

:use_py_launcher
"%PY_CMD%" -3.11 -m venv "%SCRIPT_DIR%.venv" 2>nul && goto :venv_ok
"%PY_CMD%" -m venv "%SCRIPT_DIR%.venv" 2>nul && goto :venv_ok
echo WARNING: `py` launcher could not create the virtual environment.
goto :check_python_fallback

:check_python_fallback
for /f "delims=" %%P in ('where python 2^>nul') do (
    set "PY_CMD=%%P"
    goto :use_python_direct
)
echo ERROR: Python 3.11+ is required but was not found on PATH.
goto :pauseexit

:use_python_direct
"%PY_CMD%" -m venv "%SCRIPT_DIR%.venv" 2>nul && goto :venv_ok
echo ERROR: Failed to create virtual environment with %PY_CMD%.
goto :pauseexit

:venv_ok
echo Virtual environment ready.
exit /b 0

:run
echo.
echo Resetting Python bytecode caches...
"%PWSH_EXE%" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ^
  "$root = [System.IO.Path]::GetFullPath('%SCRIPT_DIR%');" ^
  "$venv = Join-Path $root '.venv';" ^
  "Get-ChildItem -Path $root -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Where-Object { -not $_.FullName.StartsWith($venv, [System.StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop } catch { } };" ^
  "Get-ChildItem -Path $root -Recurse -File -Include '*.pyc','*.pyo' -ErrorAction SilentlyContinue | Where-Object { -not $_.DirectoryName.StartsWith($venv, [System.StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop } catch { } };"
"%PWSH_EXE%" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%SCRIPT_DIR%RunBot.ps1" %*
set "RB_EXIT=%ERRORLEVEL%"
if not "%RB_EXIT%"=="0" (
    echo.
    echo RunBot finished with exit code %RB_EXIT%.
    goto :pauseexit
)
endlocal
exit /b 0

:pauseexit
echo.
echo Exiting RunBot.
endlocal
exit /b 1
