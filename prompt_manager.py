# -*- coding: utf-8 -*-


class PromptManager:
    @staticmethod
    def _final_suffix(dataset: str) -> str:
        d = PromptManager._dataset_alias(dataset)

        if d == "gsm8k":
            return (
                "\n\n"
                "Important: put the final answer on a NEW LINE exactly as:\n"
                "#### <number>\n"
                "Do not add extra text after that line."
            )

        if d == "aime2024":
            return (
                "\n\n"
                "Important: the final answer must be a single integer from 0 to 999.\n"
                "Put the final answer on a NEW LINE exactly as:\n"
                "#### <integer>\n"
                "Examples:\n"
                "#### 7\n"
                "#### 125\n"
                "Do not write units, words, equations, boxes, tags, or any text after that line."
            )

        if d == "math500":
            return (
                "\n\n"
                "Important: put the final answer on a NEW LINE exactly as:\n"
                "\\boxed{<final expression>}\n"
                "Use a simplified final mathematical expression inside \\boxed{}.\n"
                "Do not add extra text after that line."
            )

        return (
            "\n\n"
            "Important: put the final answer on a NEW LINE exactly as:\n"
            "#### <answer>\n"
            "Do not add extra text after that line."
        )

    @staticmethod
    def build_prompt(question, strategy="baseline"):
        q = f"Question: {question}\n"

        if strategy == "prompt_baseline_cot":
            instr = "Answer: Let's think step by step."
        elif strategy == "prompt_concise_cot":
            instr = (
                "Answer: Let's think step by step, but concise. "
                "Use 4-6 short lines, each line is one equation or one variable update."
            )
        elif strategy == "prompt_compressed_cot":
            instr = (
                "Answer: Think step by step internally, then output only a compressed summary "
                "with 3-5 bullets (equations/updates only, no prose)."
            )
        elif strategy == "prompt_cod":
            instr = "Answer: Think step by step with a minimal draft (<=5 words per step)."
        elif strategy == "prompt_tokenlimit216":
            instr = "Answer: Let's think step by step, but keep reasoning within 216 tokens."
        elif strategy == "prompt_Budget-aware":
            instr = (
                "Answer: Let's think step by step. Reasoning budget: at most 6 lines; "
                "each line should be one equation or one variable update."
            )
        elif strategy in {
            "baseline",
            "early_stop",
            "confidence_stop",
            "confidence_stop_native",
            "dynamic_cot_native",
            "dynamic_cot",
            "answer_consistency",
            "es_cot",
        }:
            instr = (
                "Answer: Start with a short reasoning draft. "
                "Expand only if still uncertain."
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return q + instr +  PromptManager._final_suffix(dataset)
