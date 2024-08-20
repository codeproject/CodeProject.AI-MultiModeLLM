#!/bin/bash

# Development mode setup script ::::::::::::::::::::::::::::::::::::::::::::::
#
#                           Multi-mode LLM
#
# This script is called from the Multi-mode LLM directory using: 
#
#    bash ../../CodeProject.AI-Server/src/setup.sh
#
# The setup.sh script will find this install.sh file and execute it.
#
# For help with install scripts, notes on variables and methods available, tips,
# and explanations, see /src/modules/install_script_help.md

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../CodeProject.AI-Server/src/setup.sh"
    echo
    exit 1 
fi

if [ "${edgeDevice}" = "Raspberry Pi" ] || [ "${edgeDevice}" = "Orange Pi" ] || 
   [ "${edgeDevice}" = "Radxa ROCK" ]   || [ "${edgeDevice}" = "Jetson" ]; then
    moduleInstallErrors="Unable to install on Pi, ROCK or Jetson hardware."
fi

if [ "$moduleInstallErrors" = "" ]; then

    oneStepPIP=true  # Makes dealing with Numpy so much easier.

    HF_HUB_DISABLE_SYMLINKS_WARNING=1

    # codellama
    # sourceUrl="https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/"
    # fileToGet="codellama-7b.Q4_K_M.gguf"
      
    # Mistral
    # sourceUrl="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    # fileToGet=mistral-7b-instruct-v0.2.Q4_K_M.gguf

    # Phi-3
    sourceUrl="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/"
    fileToGet="Phi-3-mini-4k-instruct-q4.gguf"

    if [ "$verbosity" = "loud" ]; then writeLine "Looking for model: ${moduleDirPath}/models/${fileToGet}"; fi

    if [ ! -f "${moduleDirPath}/models/${fileToGet}" ]; then
        
        cacheDirPath="${downloadDirPath}/${modulesDir}/${moduleDirName}/${fileToGet}"
        
        if [ "$verbosity" = "loud" ]; then writeLine  "Looking for cache: ${cacheDirPath}"; fi
        if [ ! -f "${cacheDirPath}" ]; then
            mkdir -p "${downloadDirPath}/${modulesDir}/${moduleDirName}"
            mkdir -p "${moduleDirPath}/models"
            wget $wgetFlags -P "${downloadDirPath}/${modulesDir}/${moduleDirName}" "${sourceUrl}${fileToGet}"
        elif [ "$verbosity" = "loud" ]; then
            writeLine "File is cached" 
        fi

        if [ -f "${cacheDirPath}" ]; then 
            cp "${cacheDirPath}" "${moduleDirPath}/models/"
        fi

    else
        writeLine "${fileToGet} already downloaded." "$color_success"
    fi
    
fi