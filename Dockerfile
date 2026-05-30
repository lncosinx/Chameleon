FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装基础环境 (使用官方源)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    libgl1-mesa-glx \
    libqt5gui5 \
    && rm -rf /var/lib/apt/lists/*

# 2. 修正后的软链接命令 (添加了 -f 参数)
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /root/Chameleon

# 3. 安装 PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio

CMD ["/bin/bash"]
