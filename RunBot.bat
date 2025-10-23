@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0RunBot.ps1" %*
