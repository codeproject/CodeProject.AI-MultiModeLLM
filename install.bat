:: Installation script :::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::
::                        Multi-mode LLM
::
:: This script is only called from ..\..\CodeProject.AI-Server\src\setup.bat in
:: Dev setup, or ..\..\src\setup.bat in production
::
:: For help with install scripts, notes on variables and methods available, tips,
:: and explanations, see /src/modules/install_script_help.md

@if "%1" NEQ "install" (
    echo This script is only called from ..\..\src\setup.bat
    @pause
    @goto:eof
)

set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM Download the model file at installation so we can run without a connection to the Internet.

REM Phi-3
set sourceUrl=https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/
set fileToGet=Phi-3-mini-4k-instruct-q4.gguf

REM if not exist "!moduleDirPath!/models/!fileToGet!" (
REM     set destination=!downloadDirPath!\!modulesDir!\!moduleDirName!\!fileToGet!
REM 
REM     if not exist "!downloadDirPath!\!modulesDir!\!moduleDirName!" mkdir "!downloadDirPath!\!modulesDir!\!moduleDirName!"
REM     if not exist "!moduleDirPath!\models" mkdir "!moduleDirPath!\models"
REM 
REM     call "!utilsScript!" WriteLine "Downloading !fileToGet!" "!color_info!"
REM 
REM     powershell -command "Start-BitsTransfer -Source '!sourceUrl!!fileToGet!' -Destination '!destination!'"
REM     if errorlevel 1 (
REM         powershell -Command "Get-BitsTransfer | Remove-BitsTransfer"
REM         powershell -command "Start-BitsTransfer -Source '!sourceUrl!!fileToGet!' -Destination '!destination!'"
REM     )
REM     if errorlevel 1 (
REM         powershell -Command "Invoke-WebRequest '!sourceUrl!!fileToGet!' -OutFile '!destination!'"
REM         if errorlevel 1 (
REM             call "!utilsScript!" WriteLine "Download failed. Sorry." "!color_error!"
REM             set moduleInstallErrors=Unable to download !fileToGet!
REM         )
REM     )
REM 
REM     if exist "!destination!" (
REM         call "!utilsScript!" WriteLine "Moving !fileToGet! into the models folder." "!color_info!"
REM         move "!destination!" "!moduleDirPath!/models/" > nul
REM     ) else (
REM         call "!utilsScript!" WriteLine "Download faild. Sad face." "!color_warn!"
REM     )
REM ) else (
REM     call "!utilsScript!" WriteLine "!fileToGet! already downloaded." "!color_success!"
REM )

REM set moduleInstallErrors=
