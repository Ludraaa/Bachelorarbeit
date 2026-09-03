#!/bin/bash
#SBATCH -p addlc2_gpu-l40s
#SBATCH -J ChatKBQA_Wikidata_Port
#SBATCH -o=job.log
#SBATCH -e=job.log
#SBATCH --time=0-23:59
#SBATCH --gres=gpu:1
#SBATCH --mem=200000
##SBATCH -c 1 # number of cores
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "===== JOB START ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "====================="

source env_cluster.sh

cd Freebase-Setup

# Remove stale lock if Virtuoso is not actually responding
if [ -f virtuoso_db/virtuoso.lck ] && ! curl -sf http://localhost:3001/sparql > /dev/null 2>&1; then
    echo "Removing stale lock file..."
    rm -f virtuoso_db/virtuoso.lck
fi

# Only start if not already up
if curl -sf http://localhost:3001/sparql > /dev/null 2>&1; then
    echo "Virtuoso already running on port 3001, skipping start."
else
    nohup python virtuoso.py start 3001 -d virtuoso_db > virtuoso.log 2>&1 &
fi

cd ..

echo "===== Waiting for Virtuoso to start ====="
for i in $(seq 1 60); do
    sleep 5
    echo "--- [${i}/60] $(date) --- last 5 lines of virtuoso.log:"
    tail -5 Freebase-Setup/virtuoso_db/virtuoso.log 2>/dev/null || echo "(log not yet available)"

    if curl -sf http://localhost:3001/sparql > /dev/null 2>&1; then
        echo "Virtuoso is up after $((i * 5))s!"
        break
    fi

    if [ $i -eq 60 ]; then
        echo "ERROR: Virtuoso did not come up after 300s. Aborting."
        exit 1
    fi
done
echo "=========================================="

RUN_CONFIG=configs/runs/Freebase/CWQ/sparql.yaml

ENDPOINT_URL=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG')).get('endpoint_url', ''))")
export ENDPOINT_URL
echo "Resolved ENDPOINT_URL: $ENDPOINT_URL"

make train generate RUN_CONFIG=$RUN_CONFIG

echo "====================="
echo "End time: $(date)"
echo "===== JOB END ====="
