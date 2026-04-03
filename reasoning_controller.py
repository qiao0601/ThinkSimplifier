# -*- coding: utf-8 -*-

import re
import statistics
import time
from collections import Counter

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

    def generate(self, question, strategy="baseline"):
        prompt = PromptManager.build_prompt(question, strategy,dataset=self.dataset)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if strategy.startswith("prompt_"):
            return self._generate_full(inputs, prompt)
        if strategy == "baseline":
            return self._generate_full(inputs, prompt)

        if strategy == "answer_consistency":
            return self._answer_consistency_generation(inputs, prompt)
        if strategy == "es_cot":
            return self._es_cot_generation(inputs, prompt)
        if strategy == "dynamic_cot":
            return self._dynamic_cot_generation(question, prompt)
        if strategy == "confidence_stop":
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

    def _append_hash_answer_if_missing(self, text: str, answer: str):
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
            return "\nFinal answer (one line): \\boxed{"
        return "\nFinal answer (one line): #### "

    def _finalize_text(self, text: str):
        text = (text or "").strip()
        info = Evaluator.extract_answer_info(text, dataset=self.dataset)
        answer = info.get("answer")
        confidence_level = info.get("confidence_level")
        source_type = info.get("source_type")
        safe_sources = {"hash", "boxed", "strong_final_line"}

        # Only write back when extraction is both strict and from safe source.
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
        tail_chars = getattr(self.config, "AC_PROBE_TAIL_CHARS", 512)
        question_only = prompt.split("Answer:")[0] if "Answer:" in prompt else prompt
        reasoning_tail = (
            partial_reasoning[-tail_chars:] if len(partial_reasoning) > tail_chars else partial_reasoning
        )

        if self.dataset == "math500":
            probe_instr = (
                "Given the draft reasoning above, output only your current best final expression.\n"
                "Output exactly one line: \\boxed{<final expression>}\n"
                "\\boxed{"
            )
        elif probe_kind == "es_cot":
            probe_instr = (
                "Based on the partial reasoning above, output only your current best final answer.\n"
                "Output exactly one line: #### <answer>\n"
                "#### "
            )
        elif probe_kind == "dynamic":
            probe_instr = (
                "Give your current best final answer from this draft.\n"
                "Output only exactly one line: #### <answer>\n"
                "#### "
            )
        else:
            probe_instr = (
                "Given the draft reasoning above, output only your current best final answer.\n"
                "Output exactly one line: #### <answer>\n"
                "#### "
            )

        probe_text = f"{question_only}Answer:\n{reasoning_tail}\n\n{probe_instr}"
        probe_inputs = self.tokenizer(probe_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **probe_inputs,
                max_new_tokens=getattr(self.config, "AC_PROBE_MAX_NEW_TOKENS", 20),
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
            if not candidate.startswith("####"):
                candidate = f"#### {candidate}"
        answer = Evaluator.extract_answer(candidate, dataset=self.dataset)
        if answer is None:
            plain = decoded_new.strip()
            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", plain):
                answer = plain
        return answer

    def _answer_consistency_generation(self, inputs, prompt):

        generated = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None
        selected_token = None

        generated_text = ""
        start_time = time.time()

        boundary_chars = getattr(self.config, "AC_BOUNDARY_CHARS", ".!?\n")
        min_tokens_between = getattr(self.config, "AC_MIN_TOKENS_BETWEEN_PROBES", 24)
        k_required = getattr(self.config, "AC_K", 3)
        window_size = getattr(self.config, "AC_WINDOW_SIZE", 5)
        min_valid_probes = getattr(self.config, "AC_MIN_VALID_PROBES", 3)
        min_consecutive = getattr(self.config, "AC_MIN_CONSECUTIVE", 2)
        ac_debug = getattr(self.config, "AC_DEBUG", False)

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

            step_answer = self._probe_step_answer(prompt, generated_text, probe_kind="answer_consistency")
            if step_answer is None:
                continue

            probe_answers.append(step_answer)
            window = probe_answers[-window_size:]
            counts = Counter(window)
            majority_answer, majority_count = counts.most_common(1)[0]

            trailing =   0
            for x in reversed(probe_answers):
                if Evaluator.answers_equivalent(x, majority_answer, dataset=self.dataset):
                    trailing += 1
                else:
                    break

            if ac_debug:
                print(
                    f"[AC] step={len(probe_answers)} ans={step_answer} "
                    f"majority={majority_answer} cnt={majority_count} trailing={trailing}"
                )

            if (
                len(probe_answers) >= min_valid_probes
                and majority_count >= k_required
                and trailing >= min_consecutive
            ):
                generated_text = self._append_hash_answer_if_missing(generated_text, majority_answer)
                break

        end_time = time.time()
        generated_text = generated_text.strip()
        tokens_generated = generated.shape[1] - inputs["input_ids"].shape[1]
        return generated_text, end_time - start_time, tokens_generated
    
    #不是严谨统计检验，只是个 heuristic z-rule
    def _run_jump_test(self, d_latest: float, prev_diffs: list, dmin: float, p: float):
        if d_latest <= dmin:
            return False
        min_prev = getattr(self.config, "ES_MIN_PREV_DIFFS", 2)
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

        dmin = getattr(self.config, "ES_DMIN", 2.0)
        pval = getattr(self.config, "ES_P", 0.05)
        min_runs = getattr(self.config, "ES_MIN_RUNS", 3)
        min_step_ans = getattr(self.config, "ES_MIN_STEP_ANS", 4)
        stable_run_stop = getattr(self.config, "ES_STABLE_RUN_STOP", 4)
        min_tokens_between = getattr(self.config, "ES_MIN_TOKENS_BETWEEN_PROBES", 20)
        boundary_chars = getattr(self.config, "ES_BOUNDARY_CHARS", "\n")
        es_debug = getattr(self.config, "ES_DEBUG", False)

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

            step_answer = self._probe_step_answer(prompt, generated_text, probe_kind="es_cot")
            if step_answer is None:
                continue

            step_answers.append(step_answer)

            if last_step_answer is None:
                last_step_answer = step_answer
                current_run_len = 1
            elif Evaluator.answers_equivalent(step_answer, last_step_answer, dataset=self.dataset):
                current_run_len += 1
            else:       #如果答案变了，就结束旧段，开启新段
                run_lengths.append(current_run_len)
                last_step_answer = step_answer
                current_run_len = 1

            #run_lengths = [2, 3]    current_run_len = 2  r_hat = [2, 3, 2]
            r_hat = run_lengths + [current_run_len]

            if es_debug:
                print(
                    f"[ES] step={len(step_answers)} ans={step_answer} "
                    f"run={current_run_len} runs={run_lengths}"
                )

            if len(step_answers) >= min_step_ans and current_run_len >= stable_run_stop:
                generated_text = self._append_hash_answer_if_missing(generated_text, last_step_answer)
                break

            #第二个早停条件：run length 出现“跳跃式增长”
            if len(step_answers) >= min_step_ans and len(r_hat) >= min_runs:
                #r_hat = [1, 2, 5]  diffs = [1, 3]  最近这段连续稳定长度的增长，是否比过去明显大很多
                diffs = [r_hat[i] - r_hat[i - 1] for i in range(1, len(r_hat))]
                if diffs:       #diffs = [1, 3] diffs = [1, 3]  prev_diffs = [1]
                    d_latest = diffs[-1]
                    prev_diffs = diffs[:-1]
                    if self._run_jump_test(d_latest, prev_diffs, dmin=dmin, p=pval):
                        generated_text = self._append_hash_answer_if_missing(generated_text, step_answer)
                        break

        end_time = time.time()
        generated_text = generated_text.strip()
        tokens_generated = generated.shape[1] - inputs["input_ids"].shape[1]
        return generated_text, end_time - start_time, tokens_generated

    def _apply_tta_boost(self, logits, generated_text: str, tokens_generated: int, step_idx: int):
        if self.end_think_boost_token_id is None:
            return logits, False

        min_tokens = getattr(self.config, "TTA_MIN_TOKENS_BEFORE_BOOST", 80)
        if tokens_generated < min_tokens:
            return logits, False

        boost_interval = getattr(self.config, "TTA_BOOST_INTERVAL", 8)
        if boost_interval > 1 and (step_idx % boost_interval) != 0:
            return logits, False

        boundary_chars = getattr(self.config, "TTA_BOUNDARY_CHARS", ".!?\n")
        if not self._is_boundary(generated_text, boundary_chars):
            return logits, False

        alpha = getattr(self.config, "TTA_ALPHA", 0.9)
        max_boost = getattr(self.config, "TTA_MAX_BOOST", 6.0)

        mean_logit = logits.mean(dim=-1, keepdim=True)
        max_logit, _ = logits.max(dim=-1, keepdim=True)
        sharpness = (max_logit - mean_logit).clamp(min=0.0)
        boost = (alpha * sharpness).clamp(max=max_boost)

        boosted_logits = logits.clone()
        boosted_logits[:, self.end_think_boost_token_id] += boost.squeeze(-1)

        if getattr(self.config, "TTA_DEBUG", False):
            before = logits[0, self.end_think_boost_token_id].item()
            after = boosted_logits[0, self.end_think_boost_token_id].item()
            print(
                f"[TTA] boost step={step_idx} tokens={tokens_generated} "
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
                max_new_tokens=getattr(self.config, "TTA_COMPLETION_MAX_NEW_TOKENS", 48),
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

        start_time = time.time()
        for step_idx in range(self.config.MAX_NEW_TOKENS):
            outputs, logits = self._decode_one_token_step(
                generated=generated,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                last_token_input=selected_token,
            )
            tokens_generated_so_far = generated.shape[1] - inputs["input_ids"].shape[1]

            boosted_logits, used_tta = self._apply_tta_boost(
                logits=logits,
                generated_text=generated_text,
                tokens_generated=tokens_generated_so_far,
                step_idx=step_idx,
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

            if token_id == self.tokenizer.eos_token_id:
                stop_reason = "eos"
                break

            if self._ends_with_pattern(generated_token_ids, self.end_think_token_ids):
                stop_reason = "end_think"
                final_text, extra_completion_tokens = self._complete_final_answer_after_stop(
                    generated=generated,
                    current_text=generated_text,
                )
                break

        if final_text is None:
            final_text = generated_text.strip()

        final_text = self._finalize_text(final_text)
        end_time = time.time()

        tokens_generated = (generated.shape[1] - inputs["input_ids"].shape[1]) + extra_completion_tokens
        if getattr(self.config, "TTA_DEBUG", False):
            print(
                f"[TTA] stop_reason={stop_reason} "
                f"tokens={tokens_generated} tta_apply_count={tta_apply_count}"
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
        #考虑模型的最大上下文长度
        max_ctx = getattr(self.model.config, "max_position_embeddings", None)
        if max_ctx is None:
            tok_max = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(tok_max, int) and tok_max < 10_000_000:
                max_ctx = tok_max
        
        #可能截断 existing_text 以留出空间给新生成的 toke
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
                    exist_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
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
        """
        根据问题文本估算其复杂度，返回一个介于 0~1 之间的浮点数，用于动态调整生成阶段的停止阈值。
        最终复杂度 = 0.4×长度得分 + 0.25×数字得分 + 0.2×运算符得分 + 0.15×关键词得分，并限制在 [0,1] 内。
        """
        q = (question or "").lower()
        if not q:
            return 0.0

        # 320,20.0,12.0 等待深化到config文件形成超参
        len_score = min(len(q) / 320.0, 1.0)            #问题长度
        num_score = min(len(re.findall(r"\d", q)) / 20.0, 1.0)  #数字数量
        op_score = min(len(re.findall(r"[=+\-*/^]", q)) / 12.0, 1.0) #运算符数量
        #特定关键词的出现
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

    def _dynamic_cot_controller(self, question: str, prompt: str, text: str, stage: int):
        """
        动态 CoT 的核心控制器，用于在某个阶段评估当前生成文本的质量，决定是否应该继续生成（expand）
        question：原始问题。prompt：完整的提示文本。text：当前阶段生成的文本。stage：当前阶段编号（1 或 2）


        """
        info = Evaluator.extract_answer_info(text, dataset=self.dataset)
        strict_hit = info["answer"] is not None and info["confidence_level"] == "strict"
        has_final_line = ("####" in text) or ("\\boxed{" in text)
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

        #若严格命中答案，则调用 _probe_step_answer 进行探测，检验探测答案与提取答案是否一致
        probe_consistent = False
        probe_answer = None
        if strict_hit:
            probe_answer = self._probe_step_answer(prompt, text, probe_kind="dynamic")
            if probe_answer is not None:
                probe_consistent = Evaluator.answers_equivalent(
                    probe_answer, info["answer"], dataset=self.dataset
                )

        #计算奖励分数 reward
        reward = 0.0
        if strict_hit:
            reward += 0.50
        if has_final_line:
            reward += 0.15
        if probe_consistent:
            reward += 0.20
        if tail_complete:
            reward += 0.10
        if not has_uncertainty:
            reward += 0.10
        else:
            reward -= 0.15
        reward = max(0.0, min(1.0, reward))

        #根据问题复杂度和阶段计算动态阈值 threshold = base + alpha * complexity
        complexity = self._estimate_question_complexity(question)
        if stage == 1:
            base = getattr(self.config, "DCOT_STAGE1_BASE_THRESHOLD", 0.62)
        else:
            base = getattr(self.config, "DCOT_STAGE2_BASE_THRESHOLD", 0.72)
        threshold = min(0.95, base + getattr(self.config, "DCOT_COMPLEXITY_ALPHA", 0.18) * complexity)
        
        #若 reward < threshold，则判定需要继续扩展
        expand = reward < threshold

        return {
            "strict_hit": strict_hit,
            "answer": info["answer"],
            "probe_answer": probe_answer,
            "probe_consistent": probe_consistent,
            "has_uncertainty": has_uncertainty,
            "complexity": complexity,
            "reward": reward,
            "threshold": threshold,
            "expand": expand,
        }

    def _dynamic_cot_generation(self, question: str, prompt: str):
        """
        实现动态 CoT 策略的主流程：分阶段生成文本，每阶段后根据控制器决定是否继续，最终返回完整的生成结果。
        """
        total_time = 0.0
        total_tokens = 0

        if self.dataset == "math500":
            # Conservative budgets to avoid severe accuracy drop.
            stage1_budget = max(384, int(self.config.MAX_NEW_TOKENS * 0.60))
            stage2_budget = max(640, int(self.config.MAX_NEW_TOKENS * 0.85))
            stage3_budget = self.config.MAX_NEW_TOKENS
        else:
            stage1_budget = getattr(self.config, "DCOT_STAGE1_MAX_NEW_TOKENS", 96)
            stage2_budget = getattr(self.config, "DCOT_STAGE2_MAX_NEW_TOKENS", 192)
            stage3_budget = getattr(self.config, "DCOT_STAGE3_MAX_NEW_TOKENS", 384)
        dcot_debug = getattr(self.config, "DCOT_DEBUG", False)

        text1, time1, tok1 = self._generate_full_with_budget(prompt, stage1_budget)
        total_time += time1
        total_tokens += tok1
        ctrl1 = self._dynamic_cot_controller(question, prompt, text1, stage=1)

        if dcot_debug:
            print(f"[DCOT] stage1 tok={tok1} reward={ctrl1['reward']:.3f} th={ctrl1['threshold']:.3f}")
            print(
                f"[DCOT] strict={ctrl1['strict_hit']} probe_consistent={ctrl1['probe_consistent']} "
                f"complexity={ctrl1['complexity']:.3f} expand={ctrl1['expand']}"
            )

        # For math500, only stop at stage1 when evidence is very strong.
        if not ctrl1["expand"] and not (
            self.dataset == "math500" and not (ctrl1["strict_hit"] and ctrl1["probe_consistent"])
        ):
            return self._finalize_text(text1), total_time, total_tokens

        add2 = max(stage2_budget - stage1_budget, 1)
        text2, time2, tok2 = self._generate_continuation_with_budget(prompt, text1, add2)
        total_time += time2
        total_tokens += tok2
        ctrl2 = self._dynamic_cot_controller(question, prompt, text2, stage=2)

        if dcot_debug:
            print(f"[DCOT] stage2 tok={tok2} reward={ctrl2['reward']:.3f} th={ctrl2['threshold']:.3f}")
            print(
                f"[DCOT] strict={ctrl2['strict_hit']} probe_consistent={ctrl2['probe_consistent']} "
                f"complexity={ctrl2['complexity']:.3f} expand={ctrl2['expand']}"
            )

        if not ctrl2["expand"] and not (
            self.dataset == "math500" and not (ctrl2["strict_hit"] and ctrl2["probe_consistent"])
        ):
            return self._finalize_text(text2), total_time, total_tokens

        add3 = max(stage3_budget - stage2_budget, 1)
        text3, time3, tok3 = self._generate_continuation_with_budget(prompt, text2, add3)
        total_time += time3
        total_tokens += tok3

        if dcot_debug:
            info3 = Evaluator.extract_answer_info(text3, dataset=self.dataset)
            print(f"[DCOT] stage3 tok={tok3} answer={info3['answer']}")

        return self._finalize_text(text3), total_time, total_tokens
