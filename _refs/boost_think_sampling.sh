MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "Qwen/QwQ-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

DATASET_PATHS=(
    "../data/processed_datasets/nq_eval.jsonl"
    "../data/processed_datasets/gsm8k_eval.jsonl"
    "../data/processed_datasets/math_eval.jsonl"
    "../data/processed_datasets/gpqa_eval.jsonl"
    "../data/processed_datasets/aime_eval.jsonl"
)

# Configuration parameters
OUTPUT_PATH="../data/output/"
DATASET_LIMIT=500
TOTAL_GPUS=4  # Total number of available GPUs
OUTPUT_DIR="boost_think_sampling"

# Function to get number of required GPUs for a model
get_required_gpus() {
    local model=$1
    case $model in
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"|"deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
            echo 1  # 7B/8B models need 1 GPU
            ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
            echo 4  # 32B models need 4 GPUs
            ;;
        "Qwen/QwQ-32B")
            echo 4  # QwQ-32B models need 4 GPUs
            ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
            echo 4  # 70B model needs 4 GPUs
            ;;
        *)
            echo 1  # Default to 1 GPU
            ;;
    esac
}

# Function to find available GPUs
find_available_gpus() {
    local required_gpus=$1
    local available_gpus=()
    local free_gpu_count=0
    
    # Check each GPU
    for gpu in $(seq 0 $((TOTAL_GPUS-1))); do
        if [[ -z "${busy_gpus[$gpu]}" ]]; then
            available_gpus+=($gpu)
            ((free_gpu_count++))
            if [ ${#available_gpus[@]} -eq $required_gpus ]; then
                echo "${available_gpus[*]}" | tr ' ' ','
                return 0
            fi
        fi
    done
    
    # Not enough GPUs available
    echo "FAIL:$free_gpu_count"
    return 1
}

# Function to mark GPUs as busy
mark_gpus_busy() {
    local gpus=$1
    local gpu_list=(${gpus//,/ })
    for gpu in "${gpu_list[@]}"; do
        busy_gpus[$gpu]=1
    done
}

# Function to mark GPUs as free
mark_gpus_free() {
    local gpus=$1
    local gpu_list=(${gpus//,/ })
    for gpu in "${gpu_list[@]}"; do
        unset busy_gpus[$gpu]
    done
}

# Function to get current GPU status
get_gpu_status() {
    local status=""
    for gpu in $(seq 0 $((TOTAL_GPUS-1))); do
        if [[ -n "${busy_gpus[$gpu]}" ]]; then
            status="${status}[$gpu:BUSY] "
        else
            status="${status}[$gpu:FREE] "
        fi
    done
    echo "$status"
}

# Function to check if output file exists and is valid
is_output_file_valid() {
    local output_file=$1
    
    if [[ ! -f "$output_file" ]]; then
        return 1  # File doesn't exist
    fi
    
    # Check if file is not empty and is a valid JSON
    if [[ -s "$output_file" ]]; then
        # Try to check if the last line is valid JSON (this is a simple check)
        tail -n 1 "$output_file" | grep -q '}'
        if [ $? -eq 0 ]; then
            return 0  # File exists and appears valid
        fi
    fi
    
    return 1  # File is empty or invalid
}

# Function to run model inference
run_inference() {
    local model=$1
    local dataset_path=$2
    local output_file=$3
    local gpus=$4
    
    echo "Running $model on $dataset_path using GPUs: $gpus"
    echo "CUDA_VISIBLE_DEVICES=$gpus python ../src/sample_generation_boost_think.py $model $DATASET_LIMIT $dataset_path $output_file"
    CUDA_VISIBLE_DEVICES=$gpus python ../src/sample_generation_boost_think.py $model $DATASET_LIMIT $dataset_path $output_file
    
    # Note: We don't mark GPUs as free here anymore since function runs in background
    # That's now handled in the main loop when detecting finished processes
}

# Create output directory
mkdir -p "${OUTPUT_PATH}/${OUTPUT_DIR}/"

# Initialize arrays for task queue and busy GPUs
declare -A busy_gpus
declare -a task_queue=()
declare -A running_pids
declare -A pid_gpus  # Map PIDs to their assigned GPUs

# Counter for skipped tasks
skipped_tasks=0

# Prepare task queue
for MODEL in "${MODELS[@]}"; do
    for DATASET_PATH in "${DATASET_PATHS[@]}"; do
        MODEL_NAME_ALIAS=$(echo $MODEL | cut -d '/' -f 2)
        OUTPUT_FILE="${OUTPUT_PATH}/${OUTPUT_DIR}/${MODEL_NAME_ALIAS}_${DATASET_PATH##*/}.jsonl"
        
        # Check if output file already exists and is valid
        if is_output_file_valid "$OUTPUT_FILE"; then
            echo "Skipping task: $MODEL on $DATASET_PATH - Output file already exists"
            ((skipped_tasks++))
            continue
        fi
        
        REQUIRED_GPUS=$(get_required_gpus "$MODEL")
        
        # Add task to queue with its required number of GPUs
        task_queue+=("$MODEL|$DATASET_PATH|$OUTPUT_FILE|$REQUIRED_GPUS")
    done
done

echo "Starting task scheduler with ${#task_queue[@]} tasks in queue ($skipped_tasks tasks skipped)"
echo "GPU requirements: 7B/8B models: 1 GPU, 32B models: 4 GPUs, 70B model: 4 GPUs"
echo "Total available GPUs: $TOTAL_GPUS"

# Process tasks from queue
no_tasks_started=0
while [ ${#task_queue[@]} -gt 0 ] || [ ${#running_pids[@]} -gt 0 ]; do
    # Check for completed tasks
    completed=0
    for pid in "${!running_pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            # Task completed, release its GPUs
            assigned_gpus=${pid_gpus[$pid]}
            model_name=${running_pids[$pid]}
            
            echo "Task completed: $model_name. Releasing GPUs: $assigned_gpus"
            mark_gpus_free "$assigned_gpus"
            
            # Remove from tracking
            unset running_pids[$pid]
            unset pid_gpus[$pid]
            completed=1
        fi
    done
    
    # Print status if a task completed or if we couldn't start any tasks
    if [ $completed -eq 1 ] || [ $no_tasks_started -ge 5 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Running tasks: ${#running_pids[@]}, Queued tasks: ${#task_queue[@]}"
        echo "GPU Status: $(get_gpu_status)"
        no_tasks_started=0
    fi
    
    # Try to start new tasks if queue is not empty
    task_started=0
    if [ ${#task_queue[@]} -gt 0 ]; then
        for i in "${!task_queue[@]}"; do
            IFS='|' read -r model dataset_path output_file required_gpus <<< "${task_queue[$i]}"
            
            # Double-check if output file already exists (in case it was created after we started)
            if is_output_file_valid "$output_file"; then
                echo "Skipping task: $model on $dataset_path - Output file created since script started"
                unset task_queue[$i]
                task_queue=("${task_queue[@]}")  # Reindex array
                continue
            fi
            
            # Check if required GPUs exceeds total available
            if [[ -n "$required_gpus" && -n "$TOTAL_GPUS" && "$required_gpus" -gt "$TOTAL_GPUS" ]]; then
                echo "ERROR: Task $model requires $required_gpus GPUs, but only $TOTAL_GPUS are available in total"
                unset task_queue[$i]
                task_queue=("${task_queue[@]}")  # Reindex array
                continue
            fi
            
            # Find available GPUs for this task
            available_gpus=$(find_available_gpus $required_gpus)
            result_code=$?
            
            if [ $result_code -eq 0 ]; then
                # Mark GPUs as busy
                mark_gpus_busy "$available_gpus"
                
                echo "Starting task: $model on $dataset_path (requires $required_gpus GPUs, assigned: $available_gpus)"
                
                # Start the task
                run_inference "$model" "$dataset_path" "$output_file" "$available_gpus" &
                new_pid=$!
                
                # Store PID with its assigned GPUs and model name
                running_pids[$new_pid]="$model on $dataset_path"
                pid_gpus[$new_pid]="$available_gpus"
                
                # Remove task from queue
                unset task_queue[$i]
                task_queue=("${task_queue[@]}")  # Reindex array
                
                # Add a delay to prevent resource contention
                sleep 2
                task_started=1
                break
            else
                # Extract the number of free GPUs from the error message
                free_gpu_count=${available_gpus#FAIL:}
                
                # Only print this message occasionally to avoid flooding the console
                if [ $no_tasks_started -eq 0 ]; then
                    echo "Cannot start task: $model (requires $required_gpus GPUs, only $free_gpu_count available)"
                fi
            fi
        done
    fi
    
    # Increment counter if no tasks were started
    if [ $task_started -eq 0 ]; then
        ((no_tasks_started++))
    else
        no_tasks_started=0
    fi
    
    # Sleep briefly to prevent busy waiting
    sleep 5
done

echo "All tasks completed! ($skipped_tasks tasks were skipped because output files already existed)"

