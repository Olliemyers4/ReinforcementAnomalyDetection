@echo off  
set /p name="Enter the name of the config file: "
gutenTAG --config-yaml %name% --seed 11 --plot
pause