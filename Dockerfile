FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Set proxies if required
# ENV https_proxy= 

RUN apt remove python* -y

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
# Add sudo# RUN apt-get -y install sudo
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y --no-install-recommends python3.6

# Default to python3RUN cd /usr/bin && ln -s python3.6 python
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        imagemagick \
        ffmpeg \
        libfreetype6-dev \
        libpng-dev \
        libsm6 \
        libxext6 \
        libx11-xcb-dev \
        libglu1-mesa-dev \
        libxrender-dev \
        libzmq3-dev \
        python3.6-dev \
        python3-pip \
        python3.6-tk \
        pkg-config \
        software-properties-common \
        unzip \
        vim \
        wget \
        gedit \
        gedit-plugins \
        sudo \
        ssh \
        synaptic \
        htop \
        apt-utils

# pip3
RUN python3.6 -m pip install --upgrade \
    pip \
    setuptools

# python lib
RUN python3.6 -m pip install  --use-feature=2020-resolver \
    opencv-python \
    matplotlib \
    scipy \
    numba \
    imgaug \
    numpy \
    torch==1.4.0 \
    torchvision==0.5.0 \
    Pillow==6.2.1 \
    tqdm \
    motmetrics==1.1.3 \
    Cython \
    progress \
    easydict \
    pyquaternion \
    nuscenes-devkit \
    pyyaml \
    scikit-learn==0.22.2

RUN python3.6 -m pip install --use-feature=2020-resolver -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 

# RUN git clone --recursive https://github.com/xingyizhou/CenterTrack \
# && cd /CenterTrack/src/lib/model/networks/ \
# && git clone https://github.com/CharlesShang/DCNv2/

# RUN cd /CenterTrack/src/lib/model/networks/DCNv2/ \
# && sed -i 's/python/python3.6/g' make.sh \
# && ./make.sh

RUN git clone https://github.com/CharlesShang/DCNv2/ \
&& cd DCNv2 \
&& sed -i 's/python/python3.6/g' make.sh \
&& ./make.sh

# ==========================================
# Build ROS kinetic for python3 from source.
# ==========================================

ENV ROS_MASTER_URI=http://localhost:11311/
ENV ROS_IP=172.17.0.1

ARG DEBIAN_FRONTEND=noninteractive

# Set up timezone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ROS packages with pip3
RUN printf '\n\n Installing ROS packages with pip3.. \n\n'
RUN python3.6 -m pip install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg \
&& rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# ROS initialization
# Can ignore warning about running rosdep update with sudo because we are using sudo to install dependencies as well.
RUN printf '\n\n Initializing ROS.. \n\n'
RUN rosdep init \
&& rosdep update \
&& rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Create catkin workspace to build ROS
RUN printf '\n\n Creating catkin workspace.. \n\n'
RUN mkdir /home/ros_catkin_ws && cd /home/ros_catkin_ws \
&& catkin config --init -DCMAKE_BUILD_TYPE=Release --blacklist rqt_rviz rviz_plugin_tutorials librviz_tutorial --install

# Install ROS
RUN printf '\n\n Installing ROS kinetic.. \n\n'
RUN cd /home/ros_catkin_ws \
&& rosinstall_generator desktop_full --rosdistro kinetic --deps --tar > kinetic-desktop-full.rosinstall \
&& wstool init -j8 src kinetic-desktop-full.rosinstall \
&& rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Set up environment and dependencies
RUN printf '\n\n Setting up environment and dependencies.. \n\n'
ENV ROS_PYTHON_VERSION=3
ENV PYTHONPATH=/usr/local/lib/python3.6/dist-packages:$PYTHONPATH

RUN python3.6 -m pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04 wxPython \
&& rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

ADD install_skip.sh /home/ros_catkin_ws/install_skip.sh
RUN chmod +x /home/ros_catkin_ws/install_skip.sh
RUN cd /home/ros_catkin_ws \
&& sudo ./install_skip.sh `rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g" | sed -e "s/\<python\>/python3/g"` \
&& rosdep install --from-paths src --ignore-src -y --skip-keys="`rosdep check --from-paths src --ignore-src | grep python | sed -e "s/^apt\t//g" | sed -z "s/\n/ /g"`" \
&& find . -type f -exec sed -i 's/\/usr\/bin\/env[ ]*\<python\>/\/usr\/bin\/env python3/g' {} +

# Not sure if catkin blacklisting accepts regex, so just remove these packages entirely.
RUN cd /home/ros_catkin_ws/src \
&& rm -rf visualization_tutorials qt_gui_core rviz* rqt*

# Clone ford_msgs repo
RUN cd /home/ros_catkin_ws/src \
&& git clone https://sni21@bitbucket.org/sni21/ford_msgs.git --branch vision_golfcart

RUN ls -al /usr/bin

RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3
# https://stackoverflow.com/questions/39162622/boost-python3-missing-from-ubuntu-16-04
RUN ln -s  /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python3.so

# Build ROS
RUN printf '\n\n Building ROS.. \n\n'
RUN cd /home/ros_catkin_ws && catkin build

# clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "alias sourceros='source /home/ros_catkin_ws/install/setup.bash'" >> /root/.bashrc
RUN echo "alias symlink='ln -s /DCNv2 /CenterTrack/src/lib/model/networks/DCNv2'" >> /root/.bashrc
ENTRYPOINT /bin/bash

RUN printf '\n\n DONE. \n\n'