FROM python:3.8.0-slim-buster
MAINTAINER Alice Pagnoux (apagnoux@cisco.com)

# Input parameters
ENV INPUT_RTSP_URL="" OUTPUT_RTSP_URL="" MQTT_URL="" MQTT_TOPIC=""

# Install linux packages
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install --no-install-recommends -y  \
  libgl1-mesa-glx \
  ffmpeg \
  wget \
  python3 \
  python3-pip \
  build-essential \
  cmake \
  && rm -rf /var/lib/apt/lists/*


# Install python dependencies
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Run detection
CMD python3 test2.py