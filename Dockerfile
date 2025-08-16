# 베이스 이미지: NVIDIA CUDA와 cuDNN이 포함된 Ubuntu 20.04를 사용합니다.
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# 시스템 패키지 설치 및 환경 설정 (graphviz-dev 추가)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3.8 \
    python3.8-venv \
    python3-pip \
    wget \
    graphviz-dev && \
    rm -rf /var/lib/apt/lists/*

# Python 3.8을 기본 python3으로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 가상환경 생성 (Hailo DFC는 가상환경 내 설치를 강력히 권장)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 업그레이드
RUN pip install --upgrade pip

# 필수 라이브러리 설치 (YOLOv5 요구사항 통합 및 버전 고정)
RUN pip install \
    numpy==1.23.5 \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
    tensorflow==2.11.0 \
    matplotlib \
    opencv-python-headless \
    Pillow \
    PyYAML \
    requests \
    scipy \
    tqdm \
    pandas \
    seaborn \
    pycocotools

# Protobuf 충돌을 피하기 위해 onnx 관련 라이브러리 분리 설치
RUN pip install \
    onnx==1.13.1 \
    onnx-simplifier==0.4.19

# Hailo Dataflow Compiler 설치
COPY ./hailo_files/hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl /tmp/
RUN pip install /tmp/hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl

# Ultralytics (YOLOv5) 저장소 복제 (훈련 스크립트만 사용)
RUN git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
WORKDIR /opt/yolov5
RUN git checkout v6.1

# 작업 디렉토리 설정
WORKDIR /workspace
