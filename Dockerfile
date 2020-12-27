FROM ubuntu:18.04

WORKDIR /opt
COPY . /opt

USER root

RUN apt-get update 
RUN apt-get install -y python3.6-dev \
                       python3-pip \
                       wget \
                       gdal-bin \
                       libgdal-dev \
                       libspatialindex-dev \
                       build-essential \
                       software-properties-common \
                       apt-utils

RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y libgdal-dev
RUN pip3 install cython
RUN pip3 install --upgrade cython
RUN pip3 install setuptools
RUN pip3 install GDAL==3.0.4
RUN pip3 install numpy==1.19.1
RUN pip3 install matplotlib==3.2.1
RUN pip3 install pandas==1.0.3
RUN wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.7.1.tar.gz
RUN tar -xvf spatialindex-src-1.7.1.tar.gz
RUN cd spatialindex-src-1.7.1/ && ./configure && make && make install
RUN ldconfig                       
RUN add-apt-repository ppa:ubuntugis/ppa
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip3 install ExifRead==2.3.1
RUN pip3 install gpsphoto==2.2.3
RUN pip3 install imageio==2.8.0
RUN pip3 install networkx==2.4
RUN pip3 install opencv-python==3.4.2.16
RUN pip3 install opencv-contrib-python==3.4.2.16
RUN pip3 install piexif==1.1.3
RUN pip3 install Pillow==7.1.2
RUN pip3 install pytz==2020.1
RUN pip3 install PyWavelets==1.1.1
RUN pip3 install scikit-image==0.17.2
RUN pip3 install scikit-learn==0.22.2.post1
RUN pip3 install scipy==1.4.1
RUN pip3 install six==1.14.0
RUN pip3 install sklearn==0.0
RUN pip3 install tifffile==2020.6.3

RUN pip3 install tqdm
RUN pip3 install detecto

RUN apt-get install -y libgl1-mesa-dev

RUN pip3 install open3d
RUN pip3 install utm