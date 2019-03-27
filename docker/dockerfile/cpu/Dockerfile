# Download base image
FROM ubuntu:16.04


MAINTAINER Xiao Li <xli87@bu.edu>

# Set arguments
ARG USER=xli4217
ARG GROUP=${USER}
ARG UID=1000
ARG DIRS_TO_COPY_ROOT=dirs-to-copy

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
                                         python-tk \
                                         tmux \
                                         build-essential \             
                                         curl \
                                         libfreetype6-dev \
                                         libpng12-dev \
                                         libzmq3-dev \
                                         pkg-config \
                                         python \
                                         python-dev \
                                         python-pip \
                                         rsync \                                    
                                         software-properties-common \
                                         unzip \
                                         git \
                                         python3-tk \
                                         xvfb
                                         #rm -rf /var/lib/apt/lists/*
                                         

###################
# Install Pytorch #
###################

################
# Install Spot # 
################
# RUN wget http://www.lrde.epita.fr/dload/spot/spot-2.5.3.tar.gz && \
#     tar -xzf spot-2.5.3.tar.gz && \
#     cd spot-2.5.3 && \
#     ./configure && make && make install

# RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add - && \
#     echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
#     apt-get update && \
#     apt-get -y install spot libspot-dev spot-doc python3-spot # Or a subset of those

##########################
# Copy V-REP and env API #
##########################
COPY ${DIRS_TO_COPY_ROOT}/V-REP /home/${USER}/V-REP 
COPY ${DIRS_TO_COPY_ROOT}/cooking_env /home/${USER}/cooking_env

########
# Misc #
########
# Expose ports for Tensorboard (6006), Ipython (8888), ROS (11311)
# EXPOSE 6006 8888 11311
EXPOSE 5000-60000

WORKDIR "/home"

# Putting stuff in bashrc
RUN echo "alias em='emacs -nw'" >> /root/.bashrc
RUN echo "source /home/root/cooking_env/setup_path.sh" >> /root/.bashrc

# copy bashrc
RUN cp /root/.bashrc /home/root/

###########################
# Install python packages #
###########################
RUN pip install --upgrade pip

RUN pip --no-cache-dir install \
        fire
