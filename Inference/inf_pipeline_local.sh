# ============================================================
# Global configuration for model inference and evaluation
# ============================================================
CUDA_DEVICES_ALL="0,1,2,3,4,5,6,7"       # All GPUs available
TP_SIZE=8                                # Default tensor parallel size
MAX_LEN=4096                             # Max input sequence length
GPU_UTIL=0.9                             # Maximum GPU memory utilization
INFER_PORT=8000                          # Port for vLLM inference server
DATA_NUM=500                             # Number of data samples for inference/eval
SAMPLE_NUM=20                            # Number of samples per task
BSZ=512                                  # Batch size

ENTITY="all_entities"                    # Entity name for experiment tracking
DATA_DIR="./MACE"                        # Data directory

RESULT_DIR="./result"                    # Directory to save results
LOG_DIR="./vllm_log"                     # Directory to save logs
mkdir -p "$LOG_DIR"
PID_LOG="${LOG_DIR}/vllm_pid.log"        # Log file to record process IDs

# Model names and paths
MODEL_NAMES=(
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Qwen2.5-32B-Instruct"
  "Qwen2.5-72B-Instruct"
  "Meta-Llama-3.1-8B-Instruct"
  "Meta-Llama-3.1-70B-Instruct"
)
MODEL_PATHS=(
  "./models/Qwen-2.5-7B-instruct"
  "./models/Qwen-2.5-14B-instruct"
  "./models/Qwen-2.5-32B-instruct"
  "./models/Qwen-2.5-72B-instruct"
  "./models/Meta-Llama-3.1-8B-Instruct"
  "./models/Meta-Llama-3.1-70B-Instruct"
)

# Wait until the vLLM server is ready on the target port
wait_for_port() {
  local port=$1
  echo "⏳ Waiting for vLLM to start on port ${port}..."
  until curl -s "http://localhost:${port}/v1/models" > /dev/null; do
    sleep 5
    echo "  → Service not ready, continue waiting..."
  done
  echo "✅ vLLM service on port ${port} is ready!"
}



# Load inference model, run inference tasks, then stop process
run_infer_model() {
  local MODEL_NAME=$1
  local MODEL_PATH=$2
  local TP_SIZE
  local CUDA_DEVICES

  # Use 4 GPUs for 7B models, else use 8 GPUs
  if [[ "$MODEL_NAME" == *"7B"* ]]; then
    TP_SIZE=4
    CUDA_DEVICES="0,1,2,3"
    echo "🧩 Detected ${MODEL_NAME} → Using 4 GPUs TP=4"
  else
    TP_SIZE=8
    CUDA_DEVICES="0,1,2,3,4,5,6,7"
    echo "🧩 Detected ${MODEL_NAME} → Using 8 GPUs TP=8"
  fi

  echo "=============================="
  echo "🚀 Launching inference model: $MODEL_NAME"
  echo "=============================="

  # Start vLLM server
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size ${TP_SIZE} \
    --port ${INFER_PORT} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --dtype float16 > "${LOG_DIR}/vllm_infer_${MODEL_NAME}.log" 2>&1 &
  SERVER_PID=$!
  echo "[`date +"%Y-%m-%d %H:%M:%S"`] INFER_MODEL $MODEL_NAME PID: $SERVER_PID" >> "$PID_LOG"
  wait_for_port ${INFER_PORT}

  echo "▶️ Starting inference tasks..."
  # Run multiple inference modes
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
      --using_output_all \
      --using_host $ARGSET || true 
  done

  echo "🧹 Stopping inference model (PID=$SERVER_PID)"
}

# Run evaluation on the model using various evaluation arguments/modes
run_eval_model_api() {
  local RUN_MODEL_NAME=$1

  echo "▶️ Starting evaluation tasks..."
  for ARGSET in "--eval_only" "--using_sme" \
                "--using_latent" "--consistency_origin" \
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
      --using_output_all \
      --force_replace \
      --using_api $ARGSET || true 
  done
}

# Main loop for all models: do inference and then evaluation for each
for i in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$i]}"
  MODEL_PATH="${MODEL_PATHS[$i]}"
  run_infer_model "$MODEL_NAME" "$MODEL_PATH"
  run_eval_model_api "$MODEL_NAME"
done

echo "🎉 All model inference and evaluation completed!"
