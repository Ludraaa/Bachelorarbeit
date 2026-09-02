FROM ubuntu:22.04

LABEL maintainer="Luis Drayer <luis.drayer@tutamail.com>"

ARG UID
ARG GID

RUN groupadd -g ${GID} appuser || true
RUN useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser || true


# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && \
    apt-get install -y \
        make \
        vim \
        build-essential \
        git \
        wget \
        curl \
        python3.11 \
        python3-pip \
        python3.11-venv \
        pciutils \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Java / Apache Jena
# ============================================================

ARG JENA_VERSION=6.0.0

RUN mkdir -p /opt/ApacheJena/java && \
    wget -q \
      "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.10%2B7/OpenJDK21U-jdk_x64_linux_hotspot_21.0.10_7.tar.gz" \
      -O /tmp/jdk.tar.gz && \
    tar -xzf /tmp/jdk.tar.gz -C /opt/ApacheJena/java && \
    rm /tmp/jdk.tar.gz && \
    mv /opt/ApacheJena/java/jdk-21.0.10+7 \
       /opt/ApacheJena/java/jdk-21.0.10

RUN wget -q \
      "https://archive.apache.org/dist/jena/binaries/apache-jena-${JENA_VERSION}.tar.gz" \
      -O /tmp/jena.tar.gz && \
    tar -xzf /tmp/jena.tar.gz -C /opt/ApacheJena && \
    rm /tmp/jena.tar.gz

ENV JAVA_HOME=/opt/ApacheJena/java/jdk-21.0.10
ENV JENA_HOME=/opt/ApacheJena/apache-jena-6.0.0
ENV PATH="${JAVA_HOME}/bin:${JENA_HOME}/bin:${PATH}"

# Verify installation during build
RUN java -version && \
    riot --version


# ============================================================
# Workspace
# ============================================================

WORKDIR /workspace

COPY Makefile Makefile
COPY bashrc bashrc
COPY env.sh env.sh
COPY SETUP.md SETUP.md

COPY src/ /workspace/src

ENV PYTHONPATH=/workspace


# ============================================================
# Pipeline directories
# ============================================================

RUN mkdir -p \
    /data \
    /workspace/.hf-cache \
    /workspace/.pip-cache \
    /LLMs/data \
    /LLMs/Models \
    /LLMs/MyModels && \
    echo '{}' > /LLMs/data/dataset_info.json


# ============================================================
# Python environment
# ============================================================

COPY requirements.txt /tmp/requirements.txt

RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"


# ============================================================
# LLaMA Factory
# ============================================================
#
# Install a specific upstream commit instead of vendoring
# LLaMA Factory into the thesis repository.
#
# Commit:
# 436d26bc28b7c6422c89b63064c5a87e258ed73e
#

ARG LLAMA_FACTORY_COMMIT=436d26bc28b7c6422c89b63064c5a87e258ed73e

RUN wget -q \
      "https://github.com/hiyouga/LlamaFactory/archive/${LLAMA_FACTORY_COMMIT}.tar.gz" \
      -O /tmp/llamafactory.tar.gz && \
    mkdir /tmp/llamafactory && \
    tar -xzf /tmp/llamafactory.tar.gz \
      --strip-components=1 \
      -C /tmp/llamafactory && \
    /opt/venv/bin/pip install /tmp/llamafactory && \
    rm -rf /tmp/llamafactory /tmp/llamafactory.tar.gz


# ============================================================
# Container startup
# ============================================================

USER root

CMD ["/bin/bash", "--rcfile", "bashrc"]


# ============================================================
# Build
# ============================================================

# wharfer build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t luis-drayer-thesis .


# ============================================================
# Run
# ============================================================

# Without external data/models:
# wharfer run -it --name luis-drayer-thesis luis-drayer-thesis

# With external data:
# wharfer run -it -v /path/to/data:/data --name luis-drayer-thesis luis-drayer-thesis

# With external data, training data, and models:
# wharfer run -it -v /path/to/data:/data -v /path/to/llms-data:/LLMs/data -v /path/to/models:/LLMs/Models -v /path/to/my-models:/LLMs/MyModels --name luis-drayer-thesis luis-drayer-thesis