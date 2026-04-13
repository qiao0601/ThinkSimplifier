# -*- coding: utf-8 -*-
import argparse
import os
import re
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import Config
from evaluator import Evaluator
from metrics import Metrics
from model_loader import ModelLoader
from reasoning_controller import ReasoningController

# Put temp files somewhere with enough space.
_TMP_BASE = os.path.expanduser("~/tmp")
os.makedirs(_TMP_BASE, exist_ok=True)
os.environ["TMPDIR"] = _TMP_BASE
os.environ["TEMP"] = _TMP_BASE
os.environ["TMP"] = _TMP_BASE

INPUT_UNIT_STRATEGIES = [
    "prompt_baseline_cot",
    "prompt_concise_cot",
    "prompt_compressed_cot",
    "prompt_cod",
    "prompt_tokenlimit216",
    "prompt_Budget-aware",
]

OUTPUT_UNIT_STRATEGIES = [
    "baseline",
    "confidence_stop",
    "dynamic_cot",
    "answer_consistency",
    "es_cot",
]

INTEGRATION_COMBOS = [
    ("prompt_concise_cot", "answer_consistency"),
    ("prompt_concise_cot", "confidence_stop"),
    ("prompt_concise_cot", "dynamic_cot"),
    ("prompt_concise_cot", "es_cot"),
    ("prompt_cod", "answer_consistency"),
    ("prompt_cod", "confidence_stop"),
    ("prompt_cod", "dynamic_cot"),
    ("prompt_cod", "es_cot"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=sorted(Config.DATASET_PRESETS.keys()),
        help="Dataset alias to run.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset aliases to run sequentially.",
    )
    parser.add_argument(
        "--dataset-group",
        type=str,
        default=None,
        choices=sorted(Config.DATASET_GROUPS.keys()),
        help="Predefined dataset group alias to run sequentially.",
    )
    parser.add_argument("--split", type=str, default=None, help="HF split override.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated list. Unit names: prompt_* / baseline|confidence_stop|dynamic_cot|"
            "answer_consistency|es_cot. Integration combo format: input+output or input__output."
        ),
    )
    parser.add_argument("--input-based", action="store_true", help="Run input-based unit strategies.")
    parser.add_argument("--output-based", action="store_true", help="Run output-based unit strategies.")
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens.")
    parser.add_argument(
        "--integration-mode",
        type=str,
        default=None,
        choices=sorted(Config.SYSTEM_INTEGRATION_TEST_MODES),
        help="Override config SYSTEM_INTEGRATION_TEST_MODE.",
    )
    return parser.parse_args()


def _normalize_output_strategy(name: str):
    n = (name or "").strip()
    alias_map = {
        "answer_consistence": "answer_consistency",
        "answer_consistency": "answer_consistency",
        "confidence_stop": "confidence_stop",
        "dynamic_cot": "dynamic_cot",
        "es_cot": "es_cot",
        "baseline": "baseline",
    }
    return alias_map.get(n, n)


def _safe_strategy_name(name: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _resolve_integration_mode(args, config: Config):
    mode = args.integration_mode or getattr(config, "SYSTEM_INTEGRATION_TEST_MODE", "none")
    if mode not in Config.SYSTEM_INTEGRATION_TEST_MODES:
        raise ValueError(f"Invalid integration mode: {mode}")
    return mode


def _resolve_dataset_aliases(args, config: Config):
    if args.datasets:
        aliases = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    elif args.dataset_group:
        aliases = list(config.DATASET_GROUPS[args.dataset_group])
    elif args.dataset:
        aliases = [args.dataset]
    else:
        aliases = [config.DATASET_ALIAS]

    unknown = [a for a in aliases if a not in config.DATASET_PRESETS]
    if unknown:
        raise ValueError(f"Unknown dataset alias(es): {unknown}")
    return aliases


def _build_unit_strategy_configs(config: Config, args):
    run_input = config.RUN_INPUT_BASED
    run_output = config.RUN_OUTPUT_BASED
    if args.input_based or args.output_based:
        run_input = args.input_based
        run_output = args.output_based

    selected = [x.strip() for x in (args.strategies or "").split(",") if x.strip()]
    cfgs = []
    if selected:
        for name in selected:
            if name in INPUT_UNIT_STRATEGIES:
                cfgs.append(
                    {
                        "name": name,
                        "input_strategy": name,
                        "output_strategy": "baseline",
                        "kind": "input_unit",
                    }
                )
                continue

            out = _normalize_output_strategy(name)
            if out in OUTPUT_UNIT_STRATEGIES:
                cfgs.append(
                    {
                        "name": out,
                        "input_strategy": None,
                        "output_strategy": out,
                        "kind": "output_unit",
                    }
                )
        return cfgs

    if run_input:
        for name in INPUT_UNIT_STRATEGIES:
            cfgs.append(
                {
                    "name": name,
                    "input_strategy": name,
                    "output_strategy": "baseline",
                    "kind": "input_unit",
                }
            )

    if run_output:
        for name in OUTPUT_UNIT_STRATEGIES:
            cfgs.append(
                {
                    "name": name,
                    "input_strategy": None,
                    "output_strategy": name,
                    "kind": "output_unit",
                }
            )
    return cfgs


def _build_integration_strategy_configs(args):
    selected = [x.strip() for x in (args.strategies or "").split(",") if x.strip()]
    if not selected:
        return [
            {
                "name": f"{inp}__{out}",
                "input_strategy": inp,
                "output_strategy": out,
                "kind": "integration",
            }
            for inp, out in INTEGRATION_COMBOS
        ]

    cfgs = []
    for item in selected:
        sep = None
        if "+" in item:
            sep = "+"
        elif "__" in item:
            sep = "__"
        if sep is None:
            continue

        inp, out = item.split(sep, 1)
        inp = inp.strip()
        out = _normalize_output_strategy(out.strip())
        if inp not in INPUT_UNIT_STRATEGIES:
            continue
        if out not in OUTPUT_UNIT_STRATEGIES or out == "baseline":
            continue

        cfgs.append(
            {
                "name": f"{inp}__{out}",
                "input_strategy": inp,
                "output_strategy": out,
                "kind": "integration",
            }
        )

    if cfgs:
        return cfgs

    return [
        {
            "name": f"{inp}__{out}",
            "input_strategy": inp,
            "output_strategy": out,
            "kind": "integration",
        }
        for inp, out in INTEGRATION_COMBOS
    ]


def _extract_question_text(sample: dict, config: Config):
    if getattr(config, "QUESTION_KEYS", None):
        pieces = []
        for key in config.QUESTION_KEYS:
            v = sample.get(key, None)
            if v is None:
                continue
            v = str(v).strip()
            if v:
                pieces.append(v)
        if pieces:
            return (config.QUESTION_JOINER or "\n").join(pieces)

    if config.QUESTION_KEY and config.QUESTION_KEY in sample:
        return str(sample[config.QUESTION_KEY])

    for key in ("question", "problem", "input"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return ""


def _pick_baseline_key(results_all, strategy_configs):
    if "baseline" in results_all:
        return "baseline"

    for cfg in strategy_configs:
        if cfg["output_strategy"] == "baseline":
            name = cfg["name"]
            if name in results_all:
                return name

    keys = list(results_all.keys())
    return keys[0] if keys else None


def _run_one_dataset(config: Config, loader: ModelLoader, strategy_configs, time_str: str, phase: str):
    dataset = load_dataset(config.DATASET_NAME, split=config.SPLIT)
    if config.SAMPLE_LIMIT is not None:
        max_n = min(config.SAMPLE_LIMIT, len(dataset))
        dataset = dataset.select(range(max_n))

    controller = ReasoningController(loader.model, loader.tokenizer, config)

    suffix = f"{time_str}_{config.DATASET_ALIAS}"
    if phase != "unit":
        suffix = f"{suffix}_{phase}"
    subfolder = os.path.join(config.OUTPUT_DIR, suffix)
    os.makedirs(subfolder, exist_ok=True)

    print(f"\n================ Dataset: {config.DATASET_ALIAS} | Phase: {phase} ================")
    print(f"HF: {config.DATASET_NAME}, split={config.SPLIT}")
    print(f"Samples: {len(dataset)}")
    print("Strategy configs:")
    for cfg in strategy_configs:
        print(f"  - {cfg['name']}: input={cfg['input_strategy']} output={cfg['output_strategy']}")

    results_all = {}
    for cfg in strategy_configs:
        name = cfg["name"]
        input_strategy = cfg["input_strategy"]
        output_strategy = cfg["output_strategy"]

        print(f"\n===== Running Strategy: {name} =====")
        rows = []
        for sample in tqdm(dataset):
            question = _extract_question_text(sample, config)
            true_answer = Evaluator.extract_true_answer(sample, dataset=config.DATASET_ALIAS)

            gen_text, gen_time, gen_tokens = controller.generate(
                question=question,
                input_strategy=input_strategy,
                output_strategy=output_strategy,
            )
            pred_answer, pred_info = Evaluator.extract_answer_robust(
                gen_text,
                dataset=config.DATASET_ALIAS,
                question=question,
                config=config,
            )
            correct = Evaluator.is_correct(pred_answer, true_answer, dataset=config.DATASET_ALIAS)

            pred_answer_out = pred_answer if pred_answer is not None else "none"
            pred_confidence_out = pred_info.get("confidence_level")
            pred_source_out = pred_info.get("source_type")
            if pred_confidence_out is None or str(pred_confidence_out).strip() == "":
                pred_confidence_out = "none"
            if pred_source_out is None or str(pred_source_out).strip() == "":
                pred_source_out = "none"

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

            rows.append(
                {
                    "phase": phase,
                    "strategy_name": name,
                    "input_strategy": input_strategy if input_strategy is not None else "none",
                    "output_strategy": output_strategy,
                    "question": question,
                    "true_answer": true_answer,
                    "pred_answer": pred_answer_out,
                    "pred_confidence": pred_confidence_out,
                    "pred_source": pred_source_out,
                    "time": gen_time,
                    "tokens": gen_tokens,
                    "correct": correct,
                    "first_correct_tokens": first_correct_tokens,
                    "outcome_eff": outcome_eff,
                    "gen_text_preview": gen_text[:400].replace("\n", "\\n"),
                    "gen_text_full": gen_text,
                }
            )

        df = pd.DataFrame(rows)
        file_name = f"{_safe_strategy_name(name)}_{config.DATASET_ALIAS}.csv"
        file_path = os.path.join(subfolder, file_name)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {file_path}")

        results_all[name] = df
        print(
            f"[{name}] acc={df['correct'].mean():.4f}, "
            f"avg_tokens={df['tokens'].mean():.2f}, "
            f"avg_time={df['time'].mean():.4f}, "
            f"outcome_eff_mean={df['outcome_eff'].mean():.4f}"
        )

    baseline_key = _pick_baseline_key(results_all, strategy_configs)
    if baseline_key is None:
        return subfolder
    baseline_df = results_all[baseline_key]

    comparison_rows = []
    for cfg in strategy_configs:
        name = cfg["name"]
        metrics = Metrics.compute(baseline_df, results_all[name])
        print(f"\n===== Comparison: {name} vs {baseline_key} =====")
        for k, v in metrics.items():
            print(f"{k}: {_fmt(v)}")

        row = {
            "phase": phase,
            "strategy": name,
            "input_strategy": cfg["input_strategy"] if cfg["input_strategy"] is not None else "none",
            "output_strategy": cfg["output_strategy"],
            "baseline_strategy": baseline_key,
        }
        row.update(metrics)
        comparison_rows.append(row)

    metrics_df = pd.DataFrame(comparison_rows)
    cmp_path = os.path.join(subfolder, "comparison_metrics.csv")
    metrics_df.to_csv(cmp_path, index=False, encoding="utf-8-sig")
    print(f"Saved comparison metrics: {cmp_path}")
    return subfolder


def run_experiment(args):
    config = Config()
    if args.limit is not None and args.limit > 0:
        config.SAMPLE_LIMIT = args.limit
    if args.model:
        config.MODEL_NAME = args.model
    if args.max_new_tokens:
        config.MAX_NEW_TOKENS = args.max_new_tokens

    mode = _resolve_integration_mode(args, config)

    dataset_aliases = _resolve_dataset_aliases(args, config)
    if mode == "mainline_then_integration":
        dataset_aliases = list(config.DATASET_GROUPS["mainline"])
        

    if mode == "mainline_then_integration":
        class _UnitArgs:
            def __init__(self):
                self.strategies = None
                self.input_based = False
                self.output_based = False

        unit_strategy_configs = _build_unit_strategy_configs(config, _UnitArgs())
    else:
        unit_strategy_configs = _build_unit_strategy_configs(config, args)

    integration_strategy_configs = _build_integration_strategy_configs(args)

    if args.split and len(dataset_aliases) > 1:
        raise ValueError("--split can only be used with a single dataset.")

    phases = []
    if mode == "none":
        phases = [("unit", unit_strategy_configs)]
    elif mode == "integration_only":
        phases = [("integration", integration_strategy_configs)]
    elif mode in {"unit_and_integration", "mainline_then_integration"}:
        phases = [("unit", unit_strategy_configs), ("integration", integration_strategy_configs)]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    phases = [(n, cfgs) for (n, cfgs) in phases if cfgs]
    if not phases:
        raise ValueError("No strategy configuration resolved. Check --strategies and mode settings.")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    loader = ModelLoader(config)

    print(f"Integration mode: {mode}")
    print(f"Dataset aliases to run: {dataset_aliases}")
    for phase_name, cfgs in phases:
        print(f"Phase {phase_name}: {len(cfgs)} strategies")

    for alias in dataset_aliases:
        split_override = args.split if len(dataset_aliases) == 1 else None
        config.apply_dataset(alias, split_override=split_override)
        for phase_name, cfgs in phases:
            _run_one_dataset(
                config=config,
                loader=loader,
                strategy_configs=cfgs,
                time_str=time_str,
                phase=phase_name,
            )


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
