#!/usr/bin/bash

# Running from a pre-installed VENV

VENVDIR=$1
SCRIPT=$2

source $VENVDIR/bin/activate
python3 $SCRIPT
