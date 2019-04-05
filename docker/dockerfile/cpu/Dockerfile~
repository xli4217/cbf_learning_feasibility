# Download base image
FROM osrf/ros:kinetic-desktop-full-xenial

MAINTAINER Xiao Li <xli87@bu.edu>

# Set arguments
ARG USER=xli4217
ARG GROUP=${USER}
ARG UID=1000
ARG TENSORFLOW_ARCH=cpu
ARG TENSORFLOW_VERSION=1.5.0rc0



# RUN mkdir /etc/sudoers.d && \
#     export uid=1000 gid=1000 &&\
#     mkdir -p /home/${USER} && \
#     echo "${USER}:x:${uid}:${gid}:${USER},,,:/home/${USER}:/bin/bash" >> /etc/passwd && \
#     echo "${USER}:x:${uid}:" >> /etc/group && \
#     echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER} && \
#     chmod 0440 /etc/sudoers.d/${USER} && \   
#     chown ${uid}:${gid} -R /home/${USER}
    

#RUN useradd -r -u 1000 -g ${GROUP} ${USER}   

#USER ${USER}

# Set environment variables
ENV HOSTNAME localhost
ENV HOME /home/${USER}
ENV USER ${USER}

# Set working directory
WORKDIR "/home/${USER}"


#################
# install tools #
#################
RUN apt-get update && apt-get install -y emacs24 \
                                         terminator \
                                         wget \
                                         net-tools \
                                         mesa-utils \
                                         htop \
                                         iputils-ping \
                                         tmux \
                                         libqwt-dev
                                         
##############
# update gcc #
##############
RUN apt-get install -y  build-essential software-properties-common && \
                        add-apt-repository ppa:ubuntu-toolchain-r/test && \
                        apt-get update && \
                        apt-get install gcc-6 g++-6 -y && \
                        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
                        # apt-get update && \
                        # apt-get install gcc-snapshot -y && \
                        # apt-get install gcc-4.8 g++-4.8 -y && \
                        # update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.8;

RUN update-alternatives --config gcc

########################
# install ros packages #
########################
RUN apt-get install -y ros-kinetic-gazebo-ros-pkgs \
                       ros-kinetic-qt-build \
                       ros-kinetic-gazebo-ros-control\
                       ros-kinetic-gazebo-ros-pkgs\
                       ros-kinetic-ros-control\
                       ros-kinetic-control-toolbox\
                       ros-kinetic-realtime-tools\
                       ros-kinetic-ros-controllers\
                       ros-kinetic-xacro\
                       python-wstool\
                       ros-kinetic-tf-conversions\
                       ros-kinetic-kdl-parser\
                       ros-kinetic-moveit
                           

######################
# Install Tensorflow #
######################
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        joblib \
        plotly \
        gym \
        pydot \
        pydotplus \
        visdom \          
        # antlr4-python3-runtime \
        cloudpickle \
        scikit-optimize \
        tools \
        fire \
        # box2d \
        # ai2thor \
        networkx==1.11 \
        && \             
   python -m ipykernel.kernelspec

# RUN pip install tensorflow

################
# Install Spot # 
################
# RUN wget http://www.lrde.epita.fr/dload/spot/spot-2.5.3.tar.gz && \
#     tar -xzf spot-2.5.3.tar.gz && \
#     cd spot-2.5.3 && \
#     ./configure && make && make install

RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add - && \
    echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get -y install spot libspot-dev spot-doc python3-spot # Or a subset of those


###############
# Install Ray #
###############
RUN pip install ray


########
# Misc #
########
# Expose ports for Tensorboard (6006), Ipython (8888), ROS (11311)
# EXPOSE 6006 8888 11311
EXPOSE 5000-60000

ENV PYTHONPATH $HOME/parallel_baxter_ws/baxter_bridge:${PYTHONPATH} 

# -------------------------------------------
# dependency for baxter bridge
# -------------------------------------------
# RUN pip install zmq protobuf

# Source setup.bash
# SHELL ["/bin/bash", "-c"]
# RUN source /root/.bashrc

# Putting stuff in bashrc
RUN echo "alias em='emacs -nw'" >> /root/.bashrc && \
    echo "source $HOME/rlfps/setup_pythonpath.sh" >> /root/.bashrc && \
    echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc 

# copy bashrc
RUN cp /root/.bashrc /home/${USER}/.bashrc && \
    mkdir /home/${USER}/rlfps/

#CMD ['source', '/opt/ros/kinetic/setup.bash']
