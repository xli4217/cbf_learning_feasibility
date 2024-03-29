#!/bin/bash

OS=$1

HEADLESS=$2

if [ $OS == "mac" ]
then
    cd /home/V-REP/
    if [ $HEADLESS == "true" ]
    then
       ./vrep.app/Contents/MacOS/vrep -h -s -q -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE $RLFPS_PATH/rl_pipeline/env/vrep_env/baxter/baxter.ttt
    else
        ./vrep.app/Contents/MacOS/vrep -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE $RLFPS_PATH/rl_pipeline/env/vrep_env/baxter/baxter.ttt
    fi
elif [ $OS == "linux" ]
     cd /home/V-REP
then
    if [ $HEADLESS == "true" ]
    then
       xvfb-run ./vrep.sh -h -s -q  -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE $RLFPS_PATH/rl_pipeline/env/vrep_env/baxter/baxter.ttt
    else
       ./vrep.sh -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE $RLFPS_PATH/rl_pipeline/env/vrep_env/baxter/baxter.ttt
    fi
else
    echo "invalid OS"
fi

