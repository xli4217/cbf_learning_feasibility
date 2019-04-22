#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

# export CE_PATH="${DIR}/docker/dir-to-copy/cooking_env"
export LEARNING_PATH="${DIR}/docker/dir-to-copy/learning/"

export PYTHONPATH=$LEARNING_PATH:$PYTHONPATH

export RC_PATH=$DIR
