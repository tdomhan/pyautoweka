@echo off
set SMACMEM=1024
IF NOT "%SMAC_MEMORY%"=="" (set SMACMEM=%SMAC_MEMORY%)
set DIR=%~dp0
set EXEC=ca.ubc.cs.beta.smac.executors.AutomaticConfigurator
set jarconcat=
SETLOCAL ENABLEDELAYEDEXPANSION
for /F "delims=" %%a IN ('dir /b /s "%DIR%\*.jar"') do set jarconcat=%%a;!jarconcat!
echo Starting with %SMACMEM% MB of RAM
@echo on
java -Xmx%SMACMEM%m -cp "%DIR%conf\;%DIR%patches\;%jarconcat%%DIR%patches\ " %EXEC% %*
