#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"

#DOCKER_ROS_DIR="$(dirname "$DIR")"

DOCKER_DIR="$(dirname "$DIR")"

# this can be cpu or gpu
# ARCH=$1
ARCH="cpu"

# this  is the instance name
# INSTANCE_NAME=$2
INSTANCE_NAME='robot_cooking'

VOLUME_MAPPING=$1


IMAGE_NAME="xli4217/robot-cooking-$ARCH"


if [ "$1" == "--help" ]
then
#    echo "Usage: ./start.sh <cpu/gpu> <instance name> <volume mapping from in format host_dir:image_dir>"
    echo "Usage: ./start.sh <volume mapping from in format host_dir:image_dir>"
    exit 0
fi


xhost +local:root

if [ $ARCH == 'gpu' ] 
then
    nvidia-docker run -it  --rm \
                  --env="DISPLAY"  \
                  --env="QT_X11_NO_MITSHM=1"  \
                  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
                  --workdir="/home/$USER" \
                  --volume=$VOLUME_MAPPING \
                  --volume="/etc/group:/etc/group:ro" \
                  --volume="/etc/passwd:/etc/passwd:ro" \
                  --volume="/etc/shadow:/etc/shadow:ro" \
                  --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
                  --shm-size=8g \
                  -e LOCAL_USER_ID=`id -u $USER` \
                  -e LOCAL_GROUP_ID=`id -g $USER` \
                  -e LOCAL_GROUP_NAME=`id -gn $USER` \
                  --name $INSTANCE_NAME $IMAGE_NAME


elif [ $ARCH == 'cpu' ]
then
    docker run -it --rm \
           --env="DISPLAY"  \
           --env="QT_X11_NO_MITSHM=1"  \
           --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
           --workdir="/home/$USER" \
           --volume=$VOLUME_MAPPING \
           --volume="/etc/group:/etc/group:ro" \
           --volume="/etc/passwd:/etc/passwd:ro" \
           --volume="/etc/shadow:/etc/shadow:ro" \
           --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
           --shm-size=8g \
           -e LOCAL_USER_ID=`id -u $USER` \
           -e LOCAL_GROUP_ID=`id -g $USER` \
           -e LOCAL_GROUP_NAME=`id -gn $USER` \
           --name $INSTANCE_NAME $IMAGE_NAME
else
    echo "Architecture not found!"
fi


xhost -local:root
