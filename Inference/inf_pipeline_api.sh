DATA_NUM=500
VIEW_NUM=1000
TOP_K=5000
SAMPLE_NUM=20
BSZ=512
# Entity selection for evaluation
ENTITY="all_entities"

# Directories for data and result output
DATA_DIR="./MACE"
RESULT_DIR="./result"

# List of model names to use
MODEL_NAMES=(
  "gpt-4o-mini"
  "gpt-4o"
  "deepseek-v3"
  "gemini-2.5-flash"
)

# Run inference via API for each model
run_infer_model_api() {
  local MODEL_NAME=$1
  local MODEL_PATH=""

  echo "=============================="
  echo "🚀 Starting inference for $MODEL_NAME"
  echo "=============================="
  for ARGSET in "" "--using_sample" "--using_vanilla_verb" "--using_vanilla_verb --using_sample" \
              "--using_topk_verb" "--using_topk_verb --using_sample" "--using_post"; do
    python ./run_MLLM.py \
      --type qa_short \
      --model_path "$MODEL_PATH" \
      --batch_size $BSZ \
      --task vqa \
      --model_type llm \
      --source "$DATA_DIR" \
      --outdir "$RESULT_DIR" \
      --model_name "$MODEL_NAME" \
      --stream_output True \
      --data_num "$DATA_NUM" \
      --entity "$ENTITY" \
      --sample_num "$SAMPLE_NUM" \
      --using_api $ARGSET || true 
  done
}

# Run evaluation via API for each model
run_eval_model_api() {
  local RUN_MODEL_NAME=$1
  echo "▶️ Starting evaluation..."
  for ARGSET in "--eval_only" "--using_sme" \
                "--consistency_origin" \
                "--consistency_origin_weight_vanilla" "--consistency_origin_weight_topk"; do
    python ./run_MLLM.py \
      --type qa_short \
      --model_path "$RUN_MODEL_NAME" \
      --batch_size $BSZ \
      --task vqa \
      --model_type llm \
      --source "$DATA_DIR" \
      --outdir "$RESULT_DIR" \
      --model_name "$RUN_MODEL_NAME" \
      --stream_output True \
      --data_num "$DATA_NUM" \
      --entity "$ENTITY" \
      --using_api $ARGSET || true 
  done
}



# Main loop to process all models
for i in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$i]}"
  run_infer_model_api "$MODEL_NAME" 
  run_eval_model_api "$MODEL_NAME"
done

echo "🎉 All model inference and evaluation finished!"
