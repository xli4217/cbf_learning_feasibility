#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

PARENT="$(dirname "$DIR")"

export PYTHONPATH=$PARENT:$PYTHONPATH

export CE_PATH=$DIR
