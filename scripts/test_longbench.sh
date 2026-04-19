#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

pip install seaborn==0.13.2
pip install rouge_score==0.1.2
pip install datasets==3.2.0
pip install tqdm==4.67.3
pip install rouge==1.0.1
pip install jieba==0.42.1
pip install fuzzywuzzy==0.18.0
pip install torch==2.4.0
pip install transformers==4.46.0
pip install einops==0.8.2

models="llama3.1-8b-instruct"
tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht lcc repobench-p"

for model in $models; do
    for task in $tasks; do
        for method in full minfer flex ; do
            python -u "${PROJECT_ROOT}/eval/LongBench/pred.py" --model "$model" --task "$task" --method "$method"
        done

        for method in xattn dba; do
            for stride in 16; do
                python -u "${PROJECT_ROOT}/eval/LongBench/pred.py" --model "$model" --task "$task" --method "$method" --stride "$stride"
            done
        done
    done
done

cd "${PROJECT_ROOT}/eval/LongBench"
for model in $models; do
    python -u eval.py --model "$model"
done
