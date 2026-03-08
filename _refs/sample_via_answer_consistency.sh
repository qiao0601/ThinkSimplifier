MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "Qwen/QwQ-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

EVAL_GROUND_TRUTH_DATASETS_PATHS=(
    "../data/processed_datasets/nq_eval.jsonl"
    "../data/processed_datasets/gsm8k_eval.jsonl"
    "../data/processed_datasets/math_eval.jsonl"
    "../data/processed_datasets/gpqa_eval.jsonl"
    "../data/processed_datasets/aime_eval.jsonl"
)
 
EVAL_DATA_DIR="../data/output/partial_reasoning_eval/eval_results"
EVAL_OUTPUT_DIR=$(dirname "$EVAL_DATA_DIR")/sample_via_answer_consistency
THRESHOLD=10

mkdir -p $EVAL_OUTPUT_DIR

for MODEL in ${MODELS[@]}; do
    MODEL_NAME_ALIAS=$(echo $MODEL | cut -d '/' -f 2)
    for EVAL_GROUND_TRUTH_DATASET_PATH in ${EVAL_GROUND_TRUTH_DATASETS_PATHS[@]}; do
        EVAL_DATASET_PATH=${EVAL_DATA_DIR}/${MODEL_NAME_ALIAS}_${EVAL_GROUND_TRUTH_DATASET_PATH##*/}.jsonl
        echo "Generating sample via answer consistency for $MODEL on $EVAL_DATASET_PATH"
        echo python ../src/early_stopping_via_consistency.py --input $EVAL_DATASET_PATH --output $EVAL_OUTPUT_DIR/${MODEL_NAME_ALIAS}_${EVAL_GROUND_TRUTH_DATASET_PATH##*/}.jsonl --threshold $THRESHOLD
        python ../src/early_stopping_via_consistency.py --input $EVAL_DATASET_PATH --output $EVAL_OUTPUT_DIR/${MODEL_NAME_ALIAS}_${EVAL_GROUND_TRUTH_DATASET_PATH##*/}.jsonl --threshold $THRESHOLD
    done
done

