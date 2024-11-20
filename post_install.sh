#!/bin/bash

# Post Installation script :::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#                          Multi-mode LLM
#
# The setup.sh file will find this post_install.sh file and execute it.

if [ "$1" != "post-install" ]; then
    echo
    read -t 3 -p "This script is only called from: bash ../../CodeProject.AI-Server/src/setup.sh"
    echo
    exit 1 
fi

# Sometimes it just doesn't want to install this.
# installPythonPackagesByName "CodeProject-AI-SDK"
