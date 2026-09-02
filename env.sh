#!/bin/bash
set -a

# ============================================================
# Caches
# ============================================================

export HF_HOME=/workspace/.hf-cache
export TRANSFORMERS_CACHE=/workspace/.hf-cache
export PIP_CACHE_DIR=/workspace/.pip-cache


# ============================================================
# Java / Apache Jena
# ============================================================

export JAVA_HOME=/opt/ApacheJena/java/jdk-21.0.10
export JENA_HOME=/opt/ApacheJena/apache-jena-6.0.0


# ============================================================
# Pipeline directories
# ============================================================

export DATA_DIR=/data
export LLM_DIR=/LLMs


# ============================================================
# Network
# ============================================================

export no_proxy=localhost,127.0.0.1,0.0.0.0,tagus,10.8.152.111
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,tagus,10.8.152.111

set +a