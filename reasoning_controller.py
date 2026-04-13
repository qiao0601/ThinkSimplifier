# -*- coding: utf-8 -*-

import re
import statistics
import time

import torch

from evaluator import Evaluator
from prompt_manager import PromptManager


class ReasoningController:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataset = getattr(config, "DATASET_ALIAS", "gsm8k")

        self.end_think_token_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        self.end_think_boost_token_id = (
            self.end_think_token_ids[0] if len(self.end_think_token_ids) > 0 else None
        )

    def generate(self, question, input_strategy=None, output_strategy="baseline"):
        prompt = PromptManager.build_prompt(question, input_strategy, dataset=self.dataset)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if output_strategy == "baseline":
            return self._generate_full(inputs, prompt)
        if output_strategy == "answer_consistency":
            return self._answer_consistency_generation(inputs, prompt)
        if output_strategy == "es_cot":
            return self._es_cot_generation(inputs, prompt)
        if output_strategy == "dynamic_cot":
            return self._dynamic_cot_generation(question, prompt)
        if output_strategy == "confidence_stop":
            return self._tta_generation(inputs, prompt)

        return self._generate_full(inputs, prompt)

    def _generate_full(self, inputs, prompt):
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt) :].strip()
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        return generated, end_time - start_time, tokens_generated

    def _decode_one_token_step(self, generated, attention_mask, past_key_values, last_token_input=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=generated if past_key_values is None else last_token_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        return outputs, logits

    @staticmethod
    def _is_boundary(text: str, boundary_chars: str):
        if not text:
            return False
        tail = text.rstrip()
        if not tail:
            return False
        return tail[-1] in boundary_chars or text.endswith("\n\n")

    @staticmethod
    def _ends_with_pattern(token_ids, pattern):
        if not pattern:
            return False
        if len(token_ids) < len(pattern):
            return False
        return token_ids[-len(pattern) :] == pattern

    def _equiv(self, a, b):
        return Evaluator.answers_equivalent(a, b, dataset=self.dataset)

    def _is_validation_dataset(self):
        groups = getattr(self.config, "DATASET_GROUPS", None)
        if isinstance(groups, dict):
            validate_aliases = groups.get("validate", [])
            if self.dataset in set(validate_aliases):
                return True
        return self.dataset in {"aime2024", "amc23", "gsmhard"}

    def _mode_by_equivalence(self, values):
        if not values:
            return None, 0

        groups = []
        for v in values:
            placed = False
            for g in groups:
                if self._equiv(v, g["rep"]):
                    g["count"] += 1
                    g["last"] = v
                    placed = True
                    break
            if not placed:
                groups.append({"rep": v, "count": 1, "last": v})

        groups.sort(key=lambda x: x["count"], reverse=True)
        best = groups[0]
        return best["last"], best["count"]

    def _trailing_equiv_count(self, values, target):
        if not values or target is None:
            return 0
        n = 0
        for v in reversed(values):
            if self._equiv(v, target):
                n += 1
            else:
                break
        return n

    def _has_safe_final_marker(self, text: str):
        if self.dataset == "math500":
            return "\\boxed{" in (text or "")
        return "####" in (text or "")

    def _append_answer_if_missing(self, text: str, answer: str):
        if not text or answer is None:
            return text

        if self.dataset == "math500":
            if "\\boxed{" in text:
                return text
            return text.rstrip() + f"\n\\boxed{{{answer}}}"

        if "####" in text:
            return text
        return text.rstrip() + f"\n#### {answer}"

    def _final_answer_prefix(self):
        if self.dataset == "math500":
            return "\n\\boxed{"
        return "\n#### "

    def _finalize_text(self, text: str):
        text = (text or "").strip()
        info = Evaluator.extract_answer_info(text, dataset=self.dataset)

        answer = info.get("answer")
        confidence_level = info.get("confidence_level")
        source_type = info.get("source_type")
        safe_sources = {"hash", "boxed", "strong_final_line"}

        if answer is None:
            return text
        if confidence_level != "strict":
            return text
        if source_type not in safe_sources:
            return text

        if self.dataset == "math500":
            if "\\boxed{" not in text:
                return text + f"\n\\boxed{{{answer}}}"
            return text

        if "####" not in text:
            return text + f"\n#### {answer}"
        return text

    def _probe_step_answer(self, prompt: str, partial_reasoning: str, probe_kind: str):
        tail_chars = int(getattr(self.config, "AC_PROBE_TAIL_CHARS", 512))
        question_only = prompt.split("Answer:")[0] if "Answer:" in prompt else prompt
        reasoning_tail = partial_reasoning[-tail_chars:] if len(partial_reasoning) > tail_chars else partial_reasoning

        if self.dataset == "math500":
            probe_instr = (
                "Given the reasoning draft above, output only the current best final expression.\n"
                "Output exactly one line: \\boxed{<final expression>}\n"
                "\\boxed{"
            )
        elif probe_kind == "es_cot":
            probe_instr = (
                "Based on the reasoning draft above, output only the current best final answer.\n"
                "Output exactly one line: #### <answer>\n"
                "#### "
            )
        else:
            probe_instr = (
                "Given the reasoning draft above, output only the current best final answer.\n"
                "Output exactly one line: #### <answer>\n"
                "#### "
            )

        probe_text = f"{question_only}Answer:\n{reasoning_tail}\n\n{probe_instr}"
        probe_inputs = self.tokenizer(probe_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **probe_inputs,
                max_new_tokens=int(getattr(self.config, "AC_PROBE_MAX_NEW_TOKENS", 20)),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        in_len = probe_inputs["input_ids"].shape[1]
        gen_ids = out[0][in_len:]
        decoded_new = self.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

        candidate = decoded_new
        if self.dataset == "math500":
            if "\\boxed{" not in candidate:
                candidate = f"\\boxed{{{candidate}}}"
        else:
            if not candidate.lstrip().startswith("####"):
                candidate = f"#### {candidate}"

        info = Evaluator.extract_answer_info(candidate, dataset=self.dataset)
        answer = info.get("answer")

        if answer is None:
            plain = decoded_new.strip()
            if self.dataset == "math500":
                fallback = Evaluator._normalize_math_text(plain)
                answer = fallback
                if answer is not None:
                    info = {
                        "answer": answer,
                        "confidence_level": "weak",
                        "source_type": "probe_tail",
                    }
            else:
                if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", plain):
                    answer = plain
                    info = {
                        "answer": answer,
                        "confidence_level": "weak",
                        "source_type": "probe_tail",
                    }

        return {
            "answer": answer,
            "confidence_level": info.get("confidence_level"),
            "source_type": info.get("source_type"),
            "raw_text": decoded_new,
        }
    def _answer_consistency_generation(self, inputs, prompt):
        generated = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None
        selected_token = None

        generated_text = ""
        start_time = time.time()

        boundary_chars = getattr(self.config, "AC_BOUNDARY_CHARS", ".!?\n")
        min_tokens_between = int(getattr(self.config, "AC_MIN_TOKENS_BETWEEN_PROBES", 24))
        k_required = int(getattr(self.config, "AC_K", 3))
        window_size = int(getattr(self.config, "AC_WINDOW_SIZE", 6))
        min_valid_probes = int(getattr(self.config, "AC_MIN_VALID_PROBES", 3))
        min_consecutive = int(getattr(self.config, "AC_MIN_CONSECUTIVE", 2))
        min_progress_ratio = float(getattr(self.config, "AC_MIN_PROGRESS_RATIO", 0.35))
        if self.dataset == "math500":
            min_progress_ratio = max(min_progress_ratio, 0.55)
        require_last_probe_strict = bool(getattr(self.config, "AC_REQUIRE_LAST_PROBE_STRICT", True))
        ac_debug = bool(getattr(self.config, "AC_DEBUG", False))

        tokens_since_probe = 0
        probe_answers = []

        for _ in range(self.config.MAX_NEW_TOKENS):
            outputs, logits = self._decode_one_token_step(
                generated=generated,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                last_token_input=selected_token,
            )
            _, next_token = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            next_token = next_token.unsqueeze(-1)

            past_key_values = outputs.past_key_values
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)],
                dim=-1,
            )
            selected_token = next_token

            token_text = self.tokenizer.decode(
                next_token[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            generated_text += token_text
            tokens_since_probe += 1

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            if not self._is_boundary(generated_text, boundary_chars):
                continue
            if tokens_since_probe < min_tokens_between:
                continue
            tokens_since_probe = 0

            probe = self._probe_step_answer(prompt, generated_text, probe_kind="answer_consistency")
            step_answer = probe.get("answer")
            if step_answer is None:
                continue

            probe_answers.append(step_answer)

            window = probe_answers[-window_size:]
            majority_answer, majority_count = self._mode_by_equivalence(window)
            trailing = self._trailing_equiv_count(probe_answers, majority_answer)

            generated_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
            progress = float(generated_tokens) / float(max(self.config.MAX_NEW_TOKENS, 1))
            last_probe_strict = probe.get("confidence_level") == "strict"
            last_matches_majority = self._equiv(step_answer, majority_answer)

            if ac_debug:
                print(
                    f"[AC] probes={len(probe_answers)} ans={step_answer} "
                    f"majority={majority_answer} cnt={majority_count} trailing={trailing} "
                    f"progress={progress:.3f} strict={last_probe_strict}"
                )

            if (
                len(probe_answers) >= min_valid_probes
                and majority_count >= k_required
                and trailing >= min_consecutive
                and progress >= min_progress_ratio
                and last_matches_majority
                and (not require_last_probe_strict or last_probe_strict)
            ):
                generated_text = self._append_answer_if_missing(generated_text, majority_answer)
                break

        end_time = time.time()
        generated_text = self._finalize_text(generated_text.strip())
        tokens_generated = generated.shape[1] - inputs["input_ids"].shape[1]
        return generated_text, end_time - start_time, tokens_generated

    def _run_jump_test(self, d_latest: float, prev_diffs: list, dmin: float, p: float):
        if d_latest <= dmin:
            return False

        min_prev = int(getattr(self.config, "ES_MIN_PREV_DIFFS", 2))
        if len(prev_diffs) < min_prev:
            return False

        mu = statistics.mean(prev_diffs)
        sd = statistics.stdev(prev_diffs) if len(prev_diffs) > 1 else 0.0
        if sd == 0:
            return d_latest > mu

        if p <= 0.01:
            z = 2.326
        elif p <= 0.05:
            z = 1.645
        else:
            z = 1.282
        return d_latest > (mu + z * sd)

    def _es_cot_generation(self, inputs, prompt):
        generated = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None
        selected_token = None

        start_time = time.time()
        generated_text = ""

        step_answers = []
        run_lengths = []
        last_step_answer = None
        current_run_len = 0

        dmin = float(getattr(self.config, "ES_DMIN", 2.0))
        pval = float(getattr(self.config, "ES_P", 0.05))
        min_runs = int(getattr(self.config, "ES_MIN_RUNS", 3))
        min_step_ans = int(getattr(self.config, "ES_MIN_STEP_ANS", 4))
        min_run_after_jump = int(getattr(self.config, "ES_MIN_RUN_AFTER_JUMP", 3))
        strong_stable_run_stop = int(getattr(self.config, "ES_STRONG_STABLE_RUN_STOP", 5))
        min_progress_ratio = float(getattr(self.config, "ES_MIN_PROGRESS_RATIO", 0.50))
        if self.dataset == "math500":
            min_progress_ratio = max(min_progress_ratio, 0.60)
        min_tokens_between = int(getattr(self.config, "ES_MIN_TOKENS_BETWEEN_PROBES", 20))
        boundary_chars = getattr(self.config, "ES_BOUNDARY_CHARS", "\n")
        es_debug = bool(getattr(self.config, "ES_DEBUG", False))

        tokens_since_probe = 0

        for _ in range(self.config.MAX_NEW_TOKENS):
            outputs, logits = self._decode_one_token_step(
                generated=generated,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                last_token_input=selected_token,
            )
            _, next_token = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            next_token = next_token.unsqueeze(-1)

            past_key_values = outputs.past_key_values
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)],
                dim=-1,
            )
            selected_token = next_token

            token_text = self.tokenizer.decode(
                next_token[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            generated_text += token_text
            tokens_since_probe += 1

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            if not self._is_boundary(generated_text, boundary_chars):
                continue
            if tokens_since_probe < min_tokens_between:
                continue
            tokens_since_probe = 0

            probe = self._probe_step_answer(prompt, generated_text, probe_kind="es_cot")
            step_answer = probe.get("answer")
            if step_answer is None:
                continue

            step_answers.append(step_answer)

            if last_step_answer is None:
                last_step_answer = step_answer
                current_run_len = 1
            elif self._equiv(step_answer, last_step_answer):
                current_run_len += 1
            else:
                run_lengths.append(current_run_len)
                last_step_answer = step_answer
                current_run_len = 1

            r_hat = run_lengths + [current_run_len]
            generated_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
            progress = float(generated_tokens) / float(max(self.config.MAX_NEW_TOKENS, 1))

            if es_debug:
                print(
                    f"[ES] steps={len(step_answers)} ans={step_answer} run={current_run_len} "
                    f"runs={run_lengths} progress={progress:.3f}"
                )

            stop_by_jump = False
            if len(step_answers) >= min_step_ans and len(r_hat) >= min_runs and current_run_len >= min_run_after_jump:
                diffs = [r_hat[i] - r_hat[i - 1] for i in range(1, len(r_hat))]
                if diffs:
                    d_latest = diffs[-1]
                    prev_diffs = diffs[:-1]
                    stop_by_jump = self._run_jump_test(d_latest, prev_diffs, dmin=dmin, p=pval)

            stop_by_strong_stable = (
                len(step_answers) >= min_step_ans
                and current_run_len >= strong_stable_run_stop
                and progress >= min_progress_ratio
            )

            if stop_by_jump or stop_by_strong_stable:
                generated_text = self._append_answer_if_missing(generated_text, last_step_answer)
                break

        end_time = time.time()
        generated_text = self._finalize_text(generated_text.strip())
        tokens_generated = generated.shape[1] - inputs["input_ids"].shape[1]
        return generated_text, end_time - start_time, tokens_generated
    def _apply_tta_boost(
        self,
        logits,
        generated_text: str,
        tokens_generated: int,
        step_idx: int,
        stability_level: int,
    ):
        if self.end_think_boost_token_id is None:
            return logits, False
        if stability_level <= 0:
            return logits, False

        min_tokens = int(getattr(self.config, "TTA_MIN_TOKENS_BEFORE_BOOST", 128))
        if tokens_generated < min_tokens:
            return logits, False

        boost_interval = int(getattr(self.config, "TTA_BOOST_INTERVAL", 4))
        if boost_interval > 1 and (step_idx % boost_interval) != 0:
            return logits, False

        boundary_chars = getattr(self.config, "TTA_BOUNDARY_CHARS", ".!?\n")
        if not self._is_boundary(generated_text, boundary_chars):
            return logits, False

        alpha = float(getattr(self.config, "TTA_ALPHA", 0.4))
        max_boost = float(getattr(self.config, "TTA_MAX_BOOST", 5.0))
        level_scale = {1: 0.35, 2: 0.70, 3: 1.00}.get(stability_level, 1.00)

        mean_logit = logits.mean(dim=-1, keepdim=True)
        max_logit, _ = logits.max(dim=-1, keepdim=True)
        sharpness = (max_logit - mean_logit).clamp(min=0.0)
        boost = (alpha * sharpness * level_scale).clamp(max=max_boost)

        boosted_logits = logits.clone()
        boosted_logits[:, self.end_think_boost_token_id] += boost.squeeze(-1)

        if bool(getattr(self.config, "TTA_DEBUG", False)):
            before = logits[0, self.end_think_boost_token_id].item()
            after = boosted_logits[0, self.end_think_boost_token_id].item()
            print(
                f"[TTA] boost level={stability_level} step={step_idx} tokens={tokens_generated} "
                f"before={before:.4f} after={after:.4f}"
            )

        return boosted_logits, True

    def _complete_final_answer_after_stop(self, generated, current_text: str):
        suffix = self._final_answer_prefix()
        suffix_ids = self.tokenizer(suffix, return_tensors="pt")["input_ids"].to(self.model.device)
        completion_input_ids = torch.cat([generated, suffix_ids], dim=-1)
        completion_attention_mask = torch.ones_like(completion_input_ids, device=self.model.device)

        with torch.no_grad():
            completion_outputs = self.model.generate(
                input_ids=completion_input_ids,
                attention_mask=completion_attention_mask,
                max_new_tokens=int(getattr(self.config, "TTA_COMPLETION_MAX_NEW_TOKENS", 48)),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = completion_input_ids.shape[1]
        gen_ids = completion_outputs[0][prompt_len:]
        completion_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        final_text = current_text.rstrip() + suffix + completion_text
        extra_tokens = completion_outputs.shape[1] - completion_input_ids.shape[1]
        return final_text, extra_tokens

    def _tta_generation(self, inputs, prompt):
        generated = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None
        selected_token = None

        generated_text = ""
        generated_token_ids = []
        extra_completion_tokens = 0
        tta_apply_count = 0
        final_text = None
        stop_reason = None

        tokens_since_probe = 0
        probe_gap = int(getattr(self.config, "TTA_PROBE_GAP", 32))
        stable_min_run = int(getattr(self.config, "TTA_STABLE_MIN_RUN", 2))
        min_tokens_before_boost = int(getattr(self.config, "TTA_MIN_TOKENS_BEFORE_BOOST", 128))
        if self.dataset == "math500":
            min_tokens_before_boost = max(min_tokens_before_boost, 256)
        min_progress_ratio = float(getattr(self.config, "TTA_MIN_PROGRESS_RATIO", 0.45))
        if self.dataset == "math500":
            min_progress_ratio = max(min_progress_ratio, 0.60)
        boundary_chars = getattr(self.config, "TTA_BOUNDARY_CHARS", ".!?\n")

        stable_run = 0
        last_probe_answer = None

        start_time = time.time()
        for step_idx in range(self.config.MAX_NEW_TOKENS):
            outputs, logits = self._decode_one_token_step(
                generated=generated,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                last_token_input=selected_token,
            )

            tokens_generated_so_far = generated.shape[1] - inputs["input_ids"].shape[1]

            did_probe = False
            if (
                tokens_generated_so_far >= min_tokens_before_boost
                and tokens_since_probe >= probe_gap
                and self._is_boundary(generated_text, boundary_chars)
            ):
                did_probe = True
                probe = self._probe_step_answer(prompt, generated_text, probe_kind="confidence_stop")
                probe_answer = probe.get("answer")
                if probe_answer is not None:
                    if last_probe_answer is not None and self._equiv(probe_answer, last_probe_answer):
                        stable_run += 1
                    else:
                        stable_run = 1
                    last_probe_answer = probe_answer
                else:
                    stable_run = max(stable_run - 1, 0)

            progress = float(tokens_generated_so_far) / float(max(self.config.MAX_NEW_TOKENS, 1))
            stability_level = 0
            if stable_run >= stable_min_run and progress >= min_progress_ratio:
                if stable_run == stable_min_run:
                    stability_level = 1
                elif stable_run == stable_min_run + 1:
                    stability_level = 2
                else:
                    stability_level = 3

            boosted_logits, used_tta = self._apply_tta_boost(
                logits=logits,
                generated_text=generated_text,
                tokens_generated=tokens_generated_so_far,
                step_idx=step_idx,
                stability_level=stability_level,
            )
            if used_tta:
                tta_apply_count += 1

            _, next_token = torch.max(torch.softmax(boosted_logits, dim=-1), dim=-1)
            next_token = next_token.unsqueeze(-1)

            past_key_values = outputs.past_key_values
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)],
                dim=-1,
            )
            selected_token = next_token

            token_id = next_token.item()
            generated_token_ids.append(token_id)
            token_text = self.tokenizer.decode(
                next_token[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            generated_text += token_text

            if did_probe:
                tokens_since_probe = 0
            else:
                tokens_since_probe += 1

            if token_id == self.tokenizer.eos_token_id:
                stop_reason = "eos"
                break

            if self._ends_with_pattern(generated_token_ids, self.end_think_token_ids):
                info = Evaluator.extract_answer_info(generated_text, dataset=self.dataset)
                if (
                    info.get("answer") is not None
                    and info.get("confidence_level") == "strict"
                    and info.get("source_type") in {"hash", "boxed", "strong_final_line"}
                ):
                    stop_reason = "end_think_has_answer"
                    final_text = generated_text
                else:
                    stop_reason = "end_think"
                    final_text, extra_completion_tokens = self._complete_final_answer_after_stop(
                        generated=generated,
                        current_text=generated_text,
                    )
                break

            if stability_level >= 3 and self._has_safe_final_marker(generated_text):
                stop_reason = "stable_final_line"
                final_text = generated_text
                break

        if final_text is None:
            final_text = generated_text.strip()

        final_text = self._finalize_text(final_text)
        end_time = time.time()

        tokens_generated = (generated.shape[1] - inputs["input_ids"].shape[1]) + extra_completion_tokens
        if bool(getattr(self.config, "TTA_DEBUG", False)):
            print(
                f"[TTA] stop_reason={stop_reason} tokens={tokens_generated} "
                f"tta_apply_count={tta_apply_count} stable_run={stable_run}"
            )
        return final_text, end_time - start_time, tokens_generated
    def _generate_full_with_budget(self, prompt: str, max_new_tokens: int):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt) :].strip()
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        return generated, end_time - start_time, tokens_generated

    def _generate_continuation_with_budget(self, prompt: str, existing_text: str, add_new_tokens: int):
        full_prompt = prompt + existing_text

        max_ctx = getattr(self.model.config, "max_position_embeddings", None)
        if max_ctx is None:
            tok_max = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(tok_max, int) and tok_max < 10_000_000:
                max_ctx = tok_max

        if max_ctx is not None:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            exist_ids = self.tokenizer(existing_text, add_special_tokens=False).input_ids
            reserve = max(add_new_tokens + 8, 32)
            allow_exist = max_ctx - len(prompt_ids) - reserve
            if allow_exist < len(exist_ids):
                if allow_exist <= 0:
                    exist_ids = []
                else:
                    exist_ids = exist_ids[-allow_exist:]
                existing_text = self.tokenizer.decode(
                    exist_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                full_prompt = prompt + existing_text

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max(add_new_tokens, 1),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = full_text[len(full_prompt) :]
        combined = (existing_text + continuation).strip()
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        return combined, end_time - start_time, tokens_generated

    @staticmethod
    def _estimate_question_complexity(question: str):
        q = (question or "").lower()
        if not q:
            return 0.0

        len_score = min(len(q) / 320.0, 1.0)
        num_score = min(len(re.findall(r"\d", q)) / 20.0, 1.0)
        op_score = min(len(re.findall(r"[=+\-*/^]", q)) / 12.0, 1.0)

        keywords = [
            "probability",
            "combin",
            "geometry",
            "triangle",
            "circle",
            "prime",
            "integer",
            "sequence",
            "mod",
            "prove",
        ]
        kw_score = min(sum(1 for k in keywords if k in q) / 5.0, 1.0)
        return min(1.0, 0.4 * len_score + 0.25 * num_score + 0.2 * op_score + 0.15 * kw_score)

    def _dynamic_cot_controller(
        self,
        question: str,
        prompt: str,
        text: str,
        stage: int,
        tokens_generated: int,
        stage_budget: int,
        prev_probe_answer=None,
    ):
        info = Evaluator.extract_answer_info(text, dataset=self.dataset)
        strict_hit = info.get("answer") is not None and info.get("confidence_level") == "strict"
        safe_source = info.get("source_type") in {"hash", "boxed", "strong_final_line"}
        has_final_line = self._has_safe_final_marker(text)
        tail_complete = bool(text.strip()) and text.strip()[-1] in ".!?\n)}]"

        lower = text.lower()
        uncertainty_markers = [
            "double-check",
            "check again",
            "verify",
            "maybe",
            "not sure",
            "uncertain",
            "i think",
            "let me check",
        ]
        has_uncertainty = any(x in lower for x in uncertainty_markers)

        probe_answer = None
        probe_consistent = False
        if strict_hit or has_final_line:
            probe = self._probe_step_answer(prompt, text, probe_kind="dynamic")
            probe_answer = probe.get("answer")
            if probe_answer is not None and info.get("answer") is not None:
                probe_consistent = self._equiv(probe_answer, info.get("answer"))

        two_probe_same = (
            probe_answer is not None
            and prev_probe_answer is not None
            and self._equiv(probe_answer, prev_probe_answer)
        )

        complexity = self._estimate_question_complexity(question)
        progress = float(tokens_generated) / float(max(stage_budget, 1))

        reward = 0.0
        if strict_hit:
            reward += 0.35
        if safe_source:
            reward += 0.15
        if probe_consistent:
            reward += 0.20
        if two_probe_same:
            reward += 0.15
        if tail_complete:
            reward += 0.10
        if not has_uncertainty:
            reward += 0.05
        reward = max(0.0, min(1.0, reward))

        if stage == 1:
            min_progress = float(getattr(self.config, "DCOT_STAGE1_STOP_MIN_PROGRESS", 0.70))
            stop = (
                progress >= min_progress
                and strict_hit
                and safe_source
                and probe_consistent
                and two_probe_same
                and not has_uncertainty
            )
            base = float(getattr(self.config, "DCOT_STAGE1_BASE_THRESHOLD", 0.60))
        else:
            min_progress = float(getattr(self.config, "DCOT_STAGE2_STOP_MIN_PROGRESS", 0.50))
            strong_signals = sum(
                [
                    1 if strict_hit else 0,
                    1 if safe_source else 0,
                    1 if has_final_line else 0,
                    1 if probe_consistent else 0,
                    1 if two_probe_same else 0,
                    1 if (not has_uncertainty) else 0,
                ]
            )
            stop = progress >= min_progress and (
                (
                    strict_hit
                    and safe_source
                    and probe_consistent
                    and (two_probe_same or has_final_line)
                    and not has_uncertainty
                )
                or strong_signals >= 5
            )
            base = float(getattr(self.config, "DCOT_STAGE2_BASE_THRESHOLD", 0.72))

        threshold = min(
            0.95,
            base + float(getattr(self.config, "DCOT_COMPLEXITY_ALPHA", 0.08)) * complexity,
        )

        late_progress_gate = progress >= max(min_progress, 0.85 if stage == 1 else 0.75)
        if not stop and late_progress_gate and reward >= threshold:
            stop = True

        return {
            "strict_hit": strict_hit,
            "safe_source": safe_source,
            "answer": info.get("answer"),
            "probe_answer": probe_answer,
            "probe_consistent": probe_consistent,
            "two_probe_same": two_probe_same,
            "has_uncertainty": has_uncertainty,
            "complexity": complexity,
            "progress": progress,
            "reward": reward,
            "threshold": threshold,
            "expand": not stop,
        }

    def _dynamic_cot_generation(self, question: str, prompt: str):
        total_time = 0.0
        total_tokens = 0
        dcot_debug = bool(getattr(self.config, "DCOT_DEBUG", False))

        max_new = int(self.config.MAX_NEW_TOKENS)

        if self._is_validation_dataset():
            stage1_budget = max(int(getattr(self.config, "DCOT_STAGE1_MAX_NEW_TOKENS", 128)), int(max_new * 0.55))
            stage2_budget = max(int(getattr(self.config, "DCOT_STAGE2_MAX_NEW_TOKENS", 256)), int(max_new * 0.80))
            stage3_budget = max_new
        else:
            stage1_budget = min(int(getattr(self.config, "DCOT_STAGE1_MAX_NEW_TOKENS", 64)), max_new)
            stage2_budget = min(max(int(getattr(self.config, "DCOT_STAGE2_MAX_NEW_TOKENS", 160)), stage1_budget + 1), max_new)
            stage3_budget = min(max(int(getattr(self.config, "DCOT_STAGE3_MAX_NEW_TOKENS", 320)), stage2_budget + 1), max_new)

        text1, time1, tok1 = self._generate_full_with_budget(prompt, stage1_budget)
        total_time += time1
        total_tokens += tok1

        ctrl1 = self._dynamic_cot_controller(
            question=question,
            prompt=prompt,
            text=text1,
            stage=1,
            tokens_generated=tok1,
            stage_budget=stage1_budget,
            prev_probe_answer=None,
        )

        if dcot_debug:
            print(
                f"[DCOT] stage1 tok={tok1} progress={ctrl1['progress']:.3f} "
                f"reward={ctrl1['reward']:.3f} th={ctrl1['threshold']:.3f} expand={ctrl1['expand']}"
            )

        if not ctrl1["expand"]:
            return self._finalize_text(text1), total_time, total_tokens

        add2 = max(stage2_budget - stage1_budget, 1)
        text2, time2, tok2 = self._generate_continuation_with_budget(prompt, text1, add2)
        total_time += time2
        total_tokens += tok2

        ctrl2 = self._dynamic_cot_controller(
            question=question,
            prompt=prompt,
            text=text2,
            stage=2,
            tokens_generated=tok1 + tok2,
            stage_budget=stage2_budget,
            prev_probe_answer=ctrl1.get("probe_answer"),
        )

        if dcot_debug:
            print(
                f"[DCOT] stage2 tok={tok2} progress={ctrl2['progress']:.3f} "
                f"reward={ctrl2['reward']:.3f} th={ctrl2['threshold']:.3f} expand={ctrl2['expand']}"
            )

        if not ctrl2["expand"]:
            return self._finalize_text(text2), total_time, total_tokens

        add3 = max(stage3_budget - stage2_budget, 1)
        text3, time3, tok3 = self._generate_continuation_with_budget(prompt, text2, add3)
        total_time += time3
        total_tokens += tok3

        if dcot_debug:
            info3 = Evaluator.extract_answer_info(text3, dataset=self.dataset)
            print(f"[DCOT] stage3 tok={tok3} answer={info3.get('answer')}")

        return self._finalize_text(text3), total_time, total_tokens
