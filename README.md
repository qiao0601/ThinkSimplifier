# Reasoning Process Simplification System for Deep Thinking Models

A unified experimental framework for reproducing, evaluating, and integrating reasoning-process simplification strategies on deep-thinking language models.

This repository focuses on **efficient reasoning** for open-source deep-thinking models, using **DeepSeek-R1-Distill-Qwen-7B** as the primary target model. It provides a configurable pipeline for:

- loading datasets and models
- building prompts for multiple input-side reasoning strategies
- controlling generation with multiple output-side stopping strategies
- extracting and normalizing answers
- computing efficiency-oriented metrics
- running batch experiments across datasets and strategy combinations

The project is built for systematic comparison of **input-side methods**, **output-side methods**, and **integrated strategies** under a unified experimental setting.

---

## 1. Project Overview

Long chain-of-thought reasoning can improve accuracy on multi-step tasks, but it also increases token consumption, inference latency, and deployment cost. This repository studies how to **simplify the reasoning process** while preserving as much task performance as possible.

Instead of proposing a new training algorithm, this project provides a **systems-oriented experimental platform** for:

- reproducing representative reasoning simplification methods
- evaluating them under unified settings
- organizing them with modular interfaces
- testing integrated strategy combinations
- analyzing trade-offs between accuracy and efficiency

---

## 2. Features

- Unified experiment pipeline for reasoning simplification research
- Modular support for **input-side prompting strategies**
- Modular support for **output-side generation control strategies**
- Dataset-specific prompt formatting and answer extraction
- Batch experiment runner with CSV output
- Comparison metrics relative to a baseline strategy
- Support for both **unit strategy evaluation** and **system integration evaluation**

---

## 3. Repository Structure

```text
.
├── config.py                 # Global configuration, dataset presets, strategy hyperparameters
├── model_loader.py           # Loads tokenizer and model
├── prompt_manager.py         # Builds prompts for different input-side strategies
├── reasoning_controller.py   # Implements baseline and output-side generation control
├── evaluator.py              # Extracts answers and performs equivalence checking
├── metrics.py                # Computes accuracy/efficiency metrics
├── run_experiment.py         # Main entry for batch experiments
└── results/                  # Experiment outputs (generated after running)
````

---


## 4. Supported Datasets

The framework currently supports the following dataset aliases:

| Alias      | Hugging Face Dataset          |        Split | Question Field      | Answer Field |
| ---------- | ----------------------------- | -----------: | ------------------- | ------------ |
| `gsm8k`    | `tinyBenchmarks/tinyGSM8k`    |       `test` | `question`          | `answer`     |
| `svamp`    | `ChilleD/SVAMP`               |       `test` | `question_concat`   | `Answer`     |
| `asdiv`    | `EleutherAI/asdiv`            | `validation` | `body` + `question` | `answer`     |
| `gsmhard`  | `reasoning-machines/gsm-hard` |      `train` | `input`             | `target`     |
| `math500`  | `HuggingFaceH4/MATH-500`      |       `test` | `problem`           | `answer`     |
| `aime2024` | `HuggingFaceH4/aime_2024`     |      `train` | `problem`           | `answer`     |
| `amc23`    | `math-ai/amc23`               |       `test` | `question`          | `answer`     |

Predefined dataset groups:

* `enhance`: `gsm8k`, `svamp`, `asdiv`
* `validate`: `aime2024`, `amc23`, `gsmhard`
* `mainline`: `gsm8k`, `svamp`, `asdiv`, `aime2024`, `amc23`, `gsmhard`

---

## 5. Supported Strategies

### 5.1 Input-side strategies

These strategies modify the prompt before generation.

* `prompt_baseline_cot`
  Standard chain-of-thought prompting.

* `prompt_concise_cot`
  Encourages concise multi-step reasoning with short lines.

* `prompt_compressed_cot`
  Requests compressed reasoning summaries with equations/updates only.

* `prompt_cod`
  Chain of Draft style prompting with minimal draft steps.

* `prompt_tokenlimit216`
  Explicitly constrains reasoning length within 216 tokens.

* `prompt_Budget-aware`
  Adds a simple reasoning-budget constraint.

### 5.2 Output-side strategies

These strategies control when generation should stop.

* `baseline`
  Full generation without early stopping.

* `answer_consistency`
  Stops based on answer stability across probes.

* `es_cot`
  Early stopping logic inspired by staged convergence signals.

* `confidence_stop`
  Stops using stability-driven confidence signals.

* `dynamic_cot`
  Dynamically allocates or truncates reasoning stages.

### 5.3 Integrated strategies

The framework also supports combinations of input-side and output-side strategies.

Default integration combinations include:

* `prompt_concise_cot + answer_consistency`
* `prompt_concise_cot + confidence_stop`
* `prompt_concise_cot + dynamic_cot`
* `prompt_concise_cot + es_cot`
* `prompt_cod + answer_consistency`
* `prompt_cod + confidence_stop`
* `prompt_cod + dynamic_cot`
* `prompt_cod + es_cot`

---

## 6. Evaluation Metrics

The framework reports:

* **accuracy**
* **avg_tokens**
* **avg_time**
* **compression_ratio**
* **speedup_ratio**
* **outcome_eff_mean** (if available)
* **outcome_eff_ratio_vs_baseline** (if available)

These metrics are used to compare each strategy against a baseline result on the same dataset.

---

## 7. Environment Setup

### 7.1 Recommended Python version

Python 3.10+ is recommended.

### 7.2 Install dependencies

You can install the required packages manually. A typical environment should include:

```bash
pip install torch transformers datasets pandas tqdm openai
```

Depending on your CUDA / PyTorch environment, please install the appropriate `torch` build for your machine.

### 7.3 Model access

The default model is:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

You should make sure your environment can access Hugging Face model weights.

If needed, configure authentication for Hugging Face before running the experiments.

### 7.4 Optional API-based fallback extractor

The evaluator optionally supports a DeepSeek API fallback for answer extraction.

Set the environment variable if you want to enable it:

```bash
export DEEPSEEK_API_KEY=your_key_here
```

---

## 8. Quick Start

### 8.0 install requirements
```bash
pip install -r requirements.txt
```

### 8.1 Run one dataset with default settings

```bash
python run_experiment.py --dataset gsm8k
```

### 8.2 Run a subset of samples

```bash
python run_experiment.py --dataset gsm8k --limit 50
```

### 8.3 Run multiple datasets

```bash
python run_experiment.py --datasets gsm8k,svamp,asdiv --limit 50
```

### 8.4 Run a predefined dataset group

```bash
python run_experiment.py --dataset-group enhance --limit 50
```

### 8.5 Run only input-side unit strategies

```bash
python run_experiment.py --dataset gsm8k --input-based --limit 50
```

### 8.6 Run only output-side unit strategies

```bash
python run_experiment.py --dataset gsm8k --output-based --limit 50
```

### 8.7 Run specific strategies

```bash
python run_experiment.py --dataset gsm8k --strategies prompt_concise_cot,answer_consistency --limit 50
```

### 8.8 Run integration strategies only

```bash
python run_experiment.py --dataset gsm8k --integration-mode integration_only --limit 50
```

### 8.9 Run unit strategies first, then integration strategies

```bash
python run_experiment.py --dataset-group validate --integration-mode unit_and_integration --limit 50
```

### 8.10 Run the full mainline pipeline

```bash
python run_experiment.py --integration-mode mainline_then_integration --limit 50
```

---

## 9. Command-Line Arguments

| Argument             | Description                                         |
| -------------------- | --------------------------------------------------- |
| `--dataset`          | Run a single dataset alias                          |
| `--datasets`         | Run multiple dataset aliases separated by commas    |
| `--dataset-group`    | Run a predefined dataset group                      |
| `--split`            | Override the dataset split                          |
| `--limit`            | Limit the number of evaluated samples               |
| `--strategies`       | Specify unit strategies or integration combinations |
| `--input-based`      | Run input-side unit strategies only                 |
| `--output-based`     | Run output-side unit strategies only                |
| `--model`            | Override the model name                             |
| `--max-new-tokens`   | Override maximum generated tokens                   |
| `--integration-mode` | Choose experiment mode                              |

Supported integration modes:

* `none`
  Run unit strategies only

* `integration_only`
  Run integration strategies only

* `unit_and_integration`
  Run unit strategies first, then integration strategies

* `mainline_then_integration`
  Force mainline unit evaluation before integration evaluation

---

## 10. Prompt and Answer Formatting

The repository uses dataset-specific output constraints to make answer extraction more stable.

Examples:

* Numeric datasets such as GSM8K / SVAMP / ASDiv / GSM-Hard use:

```text
#### <number>
```

* AIME2024 and AMC23 require strict integer formatting:

```text
#### <integer>
```

* Math500 uses boxed mathematical expressions:

```text
\boxed{<final expression>}
```

This explicit formatting helps reduce answer extraction ambiguity and improves fairness across strategy comparisons.

---

## 11. How Answer Extraction Works

`evaluator.py` is designed to be robust to different output styles. It can parse:

* `#### <answer>`
* `\boxed{...}`
* `final answer: ...`
* inline final-answer expressions
* fractions and normalized mathematical forms
* bracketed or lightly formatted numeric outputs

It also performs dataset-aware equivalence checking, which is important because different strategies may output mathematically equivalent answers in different textual formats.

---

## 12. Output Files

After running experiments, results are saved under the configured output directory:

```text
./results
```

Typical outputs include:

* per-strategy CSV result files
* aggregated comparison metrics
* `comparison_metrics.csv`

Each run is timestamped, making it easier to trace configuration and experiment history.

---

## 13. Design Principles

This project follows several core design principles:

* **Unified interface**
  Different strategies are organized under one execution framework.

* **Configuration-driven design**
  Datasets, model settings, token budgets, and integration modes are controlled by configuration or CLI arguments.

* **Modularity**
  Prompt construction, generation control, answer extraction, metric computation, and experiment running are separated into different modules.

* **Reproducibility**
  The framework uses deterministic generation settings and unified answer extraction logic.

* **Extensibility**
  New datasets and strategies can be added with relatively low modification cost.

---

## 14. Typical Workflow

1. Select a dataset or dataset group
2. Select unit strategies and/or integration strategies
3. Run batch experiments with `run_experiment.py`
4. Save raw outputs and summary metrics to CSV
5. Compare strategies under the same model and decoding setting
6. Analyze trade-offs between accuracy and efficiency

---

## 15. Intended Use

This repository is primarily intended for:

* undergraduate thesis projects
* efficient reasoning experiments
* open-source reasoning model evaluation
* strategy integration studies
* reproducible benchmarking of reasoning simplification methods

It is especially suitable for experiments that need to compare multiple reasoning-control methods under a unified system implementation.

---

## 16. Future Work

Possible future extensions include:

* adding more open-source reasoning models
* supporting more datasets beyond math word problems
* expanding integration combinations
* introducing visualization scripts for experiment analysis
* adding automated ablation and parameter search tools
* improving robustness for symbolic-expression datasets

---

## 17. Citation

If you use this repository in academic work, please cite the corresponding thesis or project report.

```bibtex
@misc{reasoning_simplification_system,
  title  = {Design and Implementation of a Reasoning Process Simplification System for Deep Thinking Models},
  author = {Qiaosheng Li},
  year   = {2026},
  note   = {Undergraduate Thesis Project, Harbin Institute of Technology}
}
```

---

## 18. Acknowledgement

This project is developed as part of an undergraduate thesis on efficient reasoning for deep-thinking models, with a focus on system design, modular implementation, and multi-dataset evaluation.

---

## 19. License

Please add the appropriate open-source license for this repository if you plan to make it public.
