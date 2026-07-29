#!/bin/bash
set -a 

# caches
export HF_HOME=/work/dlclarge2/drayerl-Bachelorarbeit_SS/.hf_cache
export TRANSFORMERS_CACHE=/work/dlclarge2/drayerl-Bachelorarbeit_SS/.hf_cache
export PIP_CACHE_DIR=/work/dlclarge2/drayerl-Bachelorarbeit_SS/.pip_cache

# Path to JDK
export JAVA_HOME=ApacheJena/java/jdk-21.0.10/

# Path to Apache Jena 
export JENA_HOME=ApacheJena/apache-jena-6.0.0/

# SPARQL Endpoint
#export ENDPOINT_URL=https://qlever.dev/api/wikidata
#export ENDPOINT_URL="https://query.wikidata.org/sparql"
#export ENDPOINT_URL="http://tagus:7001/sparql" #wikidata
#export ENDPOINT_URL="http://tagus:7002/sparql" #freebase
export ENDPOINT_URL="http://localhost:3001/sparql"

export no_proxy=localhost,127.0.0.1,0.0.0.0,tagus,10.8.152.111
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,tagus,10.8.152.111

export DATA_DIR="data/"
export LLM_DIR="LLMs/"


# conda/venv
source /work/dlclarge2/drayerl-Bachelorarbeit_SS/miniconda3/bin/activate
conda activate chatkbqa
export PYTHONPATH=/work/dlclarge2/drayerl-Bachelorarbeit_SS/Bachelorarbeit/ChatKBQA
