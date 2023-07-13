FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg wget
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN apt install python-is-python3
# https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py310_23.3.1-0-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-app && \
    conda activate python-app && \
    conda install python=3.10 pip && \
    echo 'print("Hello World!")' > python-app.py

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

# If set to nothing, will install the latest version
ARG PYTORCH='2.0.1'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu118'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python -m pip uninstall -y tensorflow flax

RUN python -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python -m pip install -U "itsdangerous<2.1.0"

RUN python -m pip install poetry ipykernel


# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python setup.py develop    