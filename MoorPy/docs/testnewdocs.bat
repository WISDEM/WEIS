@echo off
title documentation test

echo %%~dp0 is "%~dp0"
call "C:/Users/shousner/Anaconda3/Scripts/activate"
cd %~dp0

call .\make html
cd build/html
call .\index.html

pause