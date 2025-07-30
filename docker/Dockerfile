# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install required packages
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    vim \
    htop \
    python3 \
    python3-pip

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install \
    matplotlib \
    jupyterlab \
    ipython \
    scipy \
    pandas \
    numpy \
    scikit-learn \
    seaborn \
    sympy \
    requests \
    pyyaml \
    tqdm \
    scikit-image 

# Set the default command to run when the container starts
CMD ["bash"]

WORKDIR /develop/code
WORKDIR /develop/data
WORKDIR /develop/results
