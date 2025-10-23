@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PWSH_EXE="

for /f "delims=" %%I in ('where pwsh 2^>nul') do (
    set "PWSH_EXE=%%I"
    goto :run
)

echo.
echo ERROR: PowerShell 7 (pwsh) is required but was not found on PATH.
echo        Install it from https://aka.ms/powershell then re-run RunBot.bat.
echo.
echo Press any key to close this window.
pause >nul
endlocal
exit /b 1

:run
"%PWSH_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%RunBot.ps1" %*
set "RB_EXIT=%ERRORLEVEL%"

if not "%RB_EXIT%"=="0" (
    echo.
    echo RunBot finished with exit code %RB_EXIT%.
    echo Press any key to close this window.
    pause >nul
)

endlocal
exit /b %RB_EXIT%
