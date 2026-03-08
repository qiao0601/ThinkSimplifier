# -*- coding: utf-8 -*-
import argparse
import os
from datetime import datetime

# put temp files somewhere with enough space
#OSError: [Errno 28] No space left on device: '/tmp/tmp1z8mfuhw' (base) [u220110602@gpu2 deepseek-ai]$
_TMP_BASE = os.path.expanduser("~/tmp")
os.makedirs(_TMP_BASE, exist_ok=True)
os.environ["TMPDIR"] = _TMP_BASE
os.environ["TEMP"] = _TMP_BASE
os.environ["TMP"] = _TMP_BASE


import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import Config
from evaluator import Evaluator
from metrics import Metrics
from model_loader import ModelLoader
from reasoning_controller import ReasoningController


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=sorted(Config.DATASET_PRESETS.keys()),
        help="Dataset alias to run.",
    )
    parser.add_argument("--split", type=str, default=None, help="HF split override.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategy list. If empty, uses group switches.",
    )
    parser.add_argument("--input-based", action="store_true", help="Run prompt-only strategies.")
    parser.add_argument("--output-based", action="store_true", help="Run output-control strategies.")
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens.")
    return parser.parse_args()


def _build_strategy_list(config: Config, args):
    prompt_strategies = [
        "prompt_baseline_cot",
        "prompt_concise_cot",
        "prompt_compressed_cot",
        "prompt_cod",
        "prompt_tokenlimit216",
        "prompt_Budget-aware",
    ]
    output_strategies = [
        "baseline",
        "confidence_stop",
        "dynamic_cot",
        "answer_consistency",
        "es_cot",
    ]

    if args.strategies:
        return [x.strip() for x in args.strategies.split(",") if x.strip()]

    run_input = config.RUN_INPUT_BASED
    run_output = config.RUN_OUTPUT_BASED
    if args.input_based or args.output_based:
        run_input = args.input_based
        run_output = args.output_based

    strategies = []
    if run_input:
        strategies.extend(prompt_strategies)
    if run_output:
        strategies.extend(output_strategies)
    if not strategies:
        raise ValueError("No strategy selected. Use --strategies or --input-based/--output-based.")
    return strategies


def run_experiment(args):
    config = Config()
    if args.dataset:
        config.apply_dataset(args.dataset, split_override=args.split)
    elif args.split:
        config.apply_dataset(config.DATASET_ALIAS, split_override=args.split)
    if args.limit is not None and args.limit > 0:
        config.SAMPLE_LIMIT = args.limit
    if args.model:
        config.MODEL_NAME = args.model
    if args.max_new_tokens:
        config.MAX_NEW_TOKENS = args.max_new_tokens

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset(config.DATASET_NAME, split=config.SPLIT)
    if config.SAMPLE_LIMIT is not None:
        max_n = min(config.SAMPLE_LIMIT, len(dataset))
        dataset = dataset.select(range(max_n))

    loader = ModelLoader(config)
    controller = ReasoningController(loader.model, loader.tokenizer, config)

    strategies = _build_strategy_list(config, args)

    results_all = {}
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = os.path.join(config.OUTPUT_DIR, f"{time_str}_{config.DATASET_ALIAS}")
    os.makedirs(subfolder, exist_ok=True)

    print(f"Dataset: {config.DATASET_ALIAS} ({config.DATASET_NAME}, split={config.SPLIT})")
    print(f"Samples: {len(dataset)}")
    print(f"Strategies: {strategies}")

    for strategy in strategies:
        print(f"\n===== Running Strategy: {strategy} =====")
        results = []
        for sample in tqdm(dataset):
            question = sample[config.QUESTION_KEY]
            true_answer = Evaluator.extract_true_answer(sample, dataset=config.DATASET_ALIAS)

            gen_text, gen_time, gen_tokens = controller.generate(
                question=question,
                strategy=strategy,
            )
            pred_answer, pred_info = Evaluator.extract_answer_robust(
                gen_text,
                dataset=config.DATASET_ALIAS,
                question=question,
                config=config,
            )
            correct = Evaluator.is_correct(pred_answer, true_answer, dataset=config.DATASET_ALIAS)

            first_correct_tokens = None
            outcome_eff = 0.0
            if gen_tokens > 0 and true_answer is not None:
                first_correct_tokens = earliest_correct_token_count(
                    gen_text=gen_text,
                    true_answer=true_answer,
                    tokenizer=controller.tokenizer,
                    dataset=config.DATASET_ALIAS,
                    step=1,
                )
                if first_correct_tokens is not None:
                    outcome_eff = float(first_correct_tokens) / float(gen_tokens)

            results.append(
                {
                    "question": question,
                    "true_answer": true_answer,
                    "pred_answer": pred_answer,
                    "pred_confidence": pred_info["confidence_level"],
                    "pred_source": pred_info["source_type"],
                    "time": gen_time,
                    "tokens": gen_tokens,
                    "correct": correct,
                    "first_correct_tokens": first_correct_tokens,
                    "outcome_eff": outcome_eff,
                    "gen_text_preview": gen_text[:400].replace("\n", "\\n"),
                    "gen_text_full": gen_text,
                }
            )

        df = pd.DataFrame(results)
        file_path = os.path.join(subfolder, f"{strategy}_{config.DATASET_ALIAS}.csv")
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {file_path}")
        results_all[strategy] = df

        print(
            f"[{strategy}] acc={df['correct'].mean():.4f}, "
            f"avg_tokens={df['tokens'].mean():.2f}, "
            f"avg_time={df['time'].mean():.4f}, "
            f"outcome_eff_mean={df['outcome_eff'].mean():.4f}"
        )

    if "baseline" in results_all:
        baseline_key = "baseline"
    elif "prompt_baseline_cot" in results_all:
        baseline_key = "prompt_baseline_cot"
    else:
        baseline_key = strategies[0]
    baseline_df = results_all[baseline_key]

    comparison_rows = []
    for strategy in strategies:
        metrics = Metrics.compute(baseline_df, results_all[strategy])
        print(f"\n===== Comparison: {strategy} vs {baseline_key} =====")
        for k, v in metrics.items():
            print(f"{k}: {_fmt(v)}")
        row = {"strategy": strategy}
        row.update(metrics)
        comparison_rows.append(row)

    metrics_df = pd.DataFrame(comparison_rows)
    file_path = os.path.join(subfolder, "comparison_metrics.csv")
    metrics_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"Saved comparison metrics: {file_path}")


def earliest_correct_token_count(gen_text: str, true_answer: str, tokenizer, dataset: str, step: int = 1):
    if not gen_text or true_answer is None:
        return None

    ids = tokenizer(gen_text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        return None

    buf_parts = []
    for i, tid in enumerate(ids, start=1):
        buf_parts.append(
            tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        )
        if i % step != 0:
            continue
        prefix_text = "".join(buf_parts)
        pred_prefix = Evaluator.extract_answer(prefix_text, dataset=dataset)
        if Evaluator.answers_equivalent(pred_prefix, true_answer, dataset=dataset):
            return i

    pred_prefix = Evaluator.extract_answer("".join(buf_parts), dataset=dataset)
    if Evaluator.answers_equivalent(pred_prefix, true_answer, dataset=dataset):
        return len(ids)
    return None


def _fmt(v):
    if v is None:
        return "None"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    run_experiment(parse_args())
