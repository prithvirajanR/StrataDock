@echo off
title StrataDock One-Click Launcher
color 0B
echo.
echo ==================================================
echo                  _         _            _         
echo      _strata_   ( )       ( )          ( )      
echo    __  _    _  _^| ^|   __  ^| ^|  _   __  ^| ^|/ )   
echo  /  _)(_ \/ _/( _ ^| / _  )^| ^|/ ) /  _)^|   /    
echo  \__ \  \ ( (_ ^| ^|^| )( (_^| ^|^|  (  \__ \^| ^|\ \    
echo  (___/   \_`\_/ \__)\__  )\_)\_\ (___/(_)\_)   
echo ==================================================
echo.
echo [1/3] Waking up Linux (WSL) and starting backend server...
:: Run the bash script directly inside WSL
start "StrataDock Server" cmd /k "wsl -- bash ~/stratadock/run.sh"

echo [2/3] Waiting for server to spin up (5 seconds)...
timeout /t 5 /nobreak >nul

echo [3/3] Opening your default web browser...
start http://localhost:8501

echo.
echo [OK] StrataDock is now running! 
echo.
echo You can minimize the new "StrataDock Server" window. 
echo To stop StrataDock, simply close that black server window.
echo.
pause
