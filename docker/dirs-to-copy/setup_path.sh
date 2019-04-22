#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

PARENT="$(dirname "$DIR")"

export PYTHONPATH=$DIR:$PYTHONPATH

export CE_PATH="${DIR}/cooking_env"
