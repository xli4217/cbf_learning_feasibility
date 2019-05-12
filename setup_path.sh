#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

# export CE_PATH="${DIR}/docker/dir-to-copy/cooking_env"
export LEARNING_PATH="${DIR}/docker/dirs-to-copy/"
export CE_PATH="${DIR}/docker/dirs-to-copy/cooking_env/"

export ROBOT_COOKING_ROS_PATH="${DIR}/src/"

export PYTHONPATH=$LEARNING_PATH:$ROBOT_COOKING_ROS_PATH:$PYTHONPATH

export RC_PATH=$DIR
