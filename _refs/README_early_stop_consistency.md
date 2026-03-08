# Early Stop via Answer Consistency Module

This module implements early stopping based on answer consistency across partial reasoning trajectories.

## Workflow

The module consists of three main steps:

1. **Collect Trajectories**: `basic_LLMs_sampling_only_answer.sh`
   - Collects model trajectories and results on all datasets
   - Generates full reasoning chains for each question

2. **Extract Partial Reasoning**: `get_partial_res_hidden_states_1gpu_eval.sh`
   - Splits trajectories into chunks (partial reasoning segments)
   - Collects model results at each chunk
   - Generates hidden states and correctness labels

3. **Generate Correctness Labels**: `eval.sh`
   - Evaluates partial reasoning results and generates correctness labels
   - Uses ground truth datasets to determine if each answer is correct
   - Outputs labeled results to `eval_results` folder

4. **Sample via Consistency**: `sample_via_answer_consistency.sh`
   - Integrates results based on answer consistency
   - Uses the correctness labels from step 3
   - Outputs final evaluation results to `sample_via_answer_consistency/` folder

## Directory Structure

```
.
├── script/
│   ├── basic_LLMs_sampling_only_answer.sh      # Step 1: Collect trajectories
│   ├── get_partial_res_hidden_states_1gpu_eval.sh  # Step 2: Extract partial reasoning
│   ├── eval.sh                                 # Step 3: Generate correctness labels
│   └── sample_via_answer_consistency.sh        # Step 4: Sample via consistency
├── src/
│   ├── sample_generation_only_answer.py        # Core generation for step 1
│   ├── get_partial_reasoning_hidden_states.py  # Extract hidden states for step 2
│   ├── get_accuracy_label.py                   # Generate correctness labels for step 3
│   ├── early_stopping_via_consistency.py       # Consistency sampling for step 4
│   └── calculate_accuracy.py                   # Calculate accuracy from results
├── chat_template/
│   └── r1_qwen.jinja                           # Chat template for model inference
└── data/
    ├── processed_datasets/                     # Input evaluation datasets
    └── output/                                 # Output directory
        ├── basic_LLMs_sampling_only_answer/    # Step 1 output
        ├── partial_reasoning_eval/             # Step 2 input/output
        │   ├── eval_results/                   # Step 3 output: Evaluation results with correctness labels
        │   └── sample_via_answer_consistency/ # Step 4 output
        └── eval_hidden_states/                 # Hidden states from step 2
```

## Usage

### Step 1: Collect Trajectories

```bash
cd script
bash basic_LLMs_sampling_only_answer.sh
```

This generates full reasoning chains and saves them to `../data/output/basic_LLMs_sampling_only_answer/`.

### Step 2: Extract Partial Reasoning

**Note**: Before running this step, you need to generate partial reasoning data from step 1's output. 
This typically requires running `collect_partial_reasoning_res.py` (not included in this module) 
to split the full reasoning into partial segments.

```bash
cd script
bash get_partial_res_hidden_states_1gpu_eval.sh
```

This processes partial reasoning data and generates hidden states and labels.

### Step 3: Generate Correctness Labels

**Important**: Before running this step, you need to set up a vLLM server for answer evaluation. The `get_accuracy_label.py` script uses an OpenAI-compatible API to evaluate answers.

1. Start a vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000
```

2. Update the API endpoint in `src/get_accuracy_label.py` if your server is running on a different host/port:
```python
openai_api_base = "http://localhost:8000/v1"  # Update this to your vLLM server endpoint
```

3. Run the evaluation:
```bash
cd script
bash eval.sh
```

This evaluates all partial reasoning results and generates correctness labels. The results are saved to `../data/output/partial_reasoning_eval/eval_results/`.

### Step 4: Sample via Consistency

```bash
cd script
bash sample_via_answer_consistency.sh
```

This integrates results based on answer consistency and outputs to `sample_via_answer_consistency/`.

### Calculate Accuracy

After step 4, you can calculate accuracy from the results:

```bash
# Calculate accuracy for a single file
python src/calculate_accuracy.py data/output/partial_reasoning_eval/sample_via_answer_consistency/MODEL_DATASET.jsonl

# Calculate accuracy for all files in a directory
python src/calculate_accuracy.py data/output/partial_reasoning_eval/sample_via_answer_consistency/

# With verbose output and grouping
python src/calculate_accuracy.py data/output/partial_reasoning_eval/sample_via_answer_consistency/ --verbose --grouped

# Save results to JSON
python src/calculate_accuracy.py data/output/partial_reasoning_eval/sample_via_answer_consistency/ --output accuracy_results.json
```

## Configuration

Edit the shell scripts to modify:
- Model list
- Dataset paths
- GPU configuration
- Output directories
- Threshold for consistency (in `sample_via_answer_consistency.sh`)

## Dependencies

- vllm
- transformers
- torch
- tqdm
- numpy
- nltk (for sentence tokenization)

## Output Format

The final output files contain JSONL format with the following fields:
- `question`: The input question
- `answer`: The extracted answer
- `correctness`: Label indicating correctness (1=correct, 0=incorrect, -1=not evaluated)
- `partial_reasoning`: The partial reasoning segment used
- `generated_text`: The full generated text

## Accuracy Calculation

The `calculate_accuracy.py` script calculates:
- Accuracy = (number of correct answers) / (total evaluated answers) × 100%
- Excludes samples with `correctness = -1` (not evaluated) from the denominator

