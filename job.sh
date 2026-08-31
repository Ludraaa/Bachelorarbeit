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

source env.sh


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

# one time adaption
#python src/temp/adapt_chatkbqa_pred.py --dataset ../OriginalChatKBQA/data/CWQ/generation/merged/CWQ_test.json --preds ../OriginalChatKBQA/Reading/LLaMA-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation_beam/generated_predictions.jsonl --output data/temp.json --kb freebase --type-map ../OriginalChatKBQA/data/CWQ/generation/label_maps/CWQ_train_type_label_map.json
# result comparison
#python src/temp/compare_results.py \
#    data/WebQSP/predictions/Llama-2-7b_WebQSP-sparql/evaluated/ChatKBQA.type_map+ChatKBQA.facc1+ChatKBQA.simple+ChatKBQA.neighborhood/WebQSP_test.sparql.json \
#    data/WQSP_qlever/predictions/Qwen25-7b_WQSP_qlever-sparql/evaluated/ChatKBQA.type_map+ChatKBQA.facc1+ChatKBQA.simple+ChatKBQA.neighborhood/WQSP_qlever_test.sparql.json --by question --a_better_only
#python src/temp/training_data_check.py --input data/WQSP_qlever/generation/merged/WQSP_qlever_test.sparql.json --kb freebase --endpoint http://tagus:7002/sparql



#python src/sparql_to_sexpr.py --dataset CWQ_qlever --mode sparql --kb freebase --config configs/datasets/CWQ.yaml

#python src/insert_labels.py --dataset CWQ_qlever --split test --kb freebase --debug

#python src/prepare_llm_data.py --dataset CWQ_qlever --split train

#lmf train configs/training/wqsp_chatkbqa.yaml

#python src/generate_predictions.py --config configs/infer/llama.yaml --dataset CWQ --split test --mode sparql --kb freebase --diversity_penalty 1.0 --num_beams 8 #--max_samples 15


python src/resolve_predictions.py \
    --dataset CWQ --split test --mode sparql \
    --model_id Llama-2-13b_CWQ-sparql \
    --entity_linkers ChatKBQA.type_map,ChatKBQA.gold_exact,ChatKBQA.gold_simcse,ChatKBQA.facc1 \
    --predicate_linkers ChatKBQA.simple,ChatKBQA.neighborhood \
    --kb freebase \
    --k1_per_pass 500,30 \
    --k2_per_pass 1,300 \
    --linker_params '{}' \
    --debug \
    --note "official run" \
    --beam_limits 8
#    #--item_time_limit_sec 1800
#    #--max_samples 50

python src/eval_predictions.py \
    --dataset CWQ --split test --mode sparql \
    --model_id Llama-2-13b_CWQ-sparql \
    --entity_linkers ChatKBQA.type_map,ChatKBQA.gold_exact,ChatKBQA.gold_simcse,ChatKBQA.facc1 \
    --predicate_linkers ChatKBQA.simple,ChatKBQA.neighborhood \
    --get-live-gold

echo "====================="
echo "End time: $(date)"
echo "===== JOB END ====="
