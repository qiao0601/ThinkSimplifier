# Boost Think Sampling Module

This module implements the boost think sampling method for generating model outputs with reasoning.

## Directory Structure

```
.
├── script/
│   └── boost_think_sampling.sh    # Main execution script
├── src/
│   ├── sample_generation_boost_think.py  # Core generation script
│   ├── math_util.py                # Math utility functions
│   └── math_equivalence.py         # Math equivalence checking
└── data/
    ├── processed_datasets/         # Input evaluation datasets
    │   ├── nq_eval.jsonl
    │   ├── gsm8k_eval.jsonl
    │   ├── math_eval.jsonl
    │   ├── gpqa_eval.jsonl
    │   └── aime_eval.jsonl
    └── output/                     # Output directory for results
        └── boost_think_sampling/
```

## Usage

Run the script from the `script/` directory:

```bash
cd script
bash boost_think_sampling.sh
```

The script will:
1. Load models and datasets as specified in the script
2. Run inference with boost think sampling
3. Save results to `../data/output/boost_think_sampling/`

## Configuration

Edit `boost_think_sampling.sh` to modify:
- Model list
- Dataset paths
- GPU configuration
- Output directory

## Dependencies

- vllm
- transformers
- datasets
- tqdm
- numpy

