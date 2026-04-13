# -*- coding: utf-8 -*-
import json
import re


class Evaluator:
    _NUMERIC_DATASETS = {"gsm8k", "svamp", "asdiv", "gsmhard", "aime2024", "amc23"}

    _NUM_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")
    _HASH_RE = re.compile(r"####\s*([^\n\r]+)")
    _STRONG_FINAL_RE = re.compile(
        r"^\s*(?:the\s+)?final\s+answer(?:\s+is)?\s*[:：]?\s*([^\n\r]+)\s*$",
        flags=re.IGNORECASE,
    )
    _STRONG_FINAL_SHORT_RE = re.compile(
        r"^\s*final\s*[:：]\s*([^\n\r]+)\s*$",
        flags=re.IGNORECASE,
    )
    _STRONG_FINAL_INLINE_RE = re.compile(
        r"(?:the\s+)?final\s+answer(?:\s+is)?\s*[:：]?\s*([^\n\r]+)",
        flags=re.IGNORECASE,
    )
    _FRAC_RE = re.compile(r"\\(?:frac|dfrac|tfrac)\{([^{}]+)\}\{([^{}]+)\}")

    @staticmethod
    def _dataset_alias(dataset: str):
        d = (dataset or "gsm8k").strip().lower()
        if d in {"aime", "aime2024"}:
            return "aime2024"
        if d in {"amc", "amc23", "amc_23"}:
            return "amc23"
        if d in {"math", "math500"}:
            return "math500"
        if d in {"gsmhard", "gsm-hard", "gsm_hard"}:
            return "gsmhard"
        if d in {"svamp", "asdiv"}:
            return d
        return "gsm8k"

    @staticmethod
    def _clean_text_for_extraction(text: str):
        if not text:
            return ""
        t = str(text)
        t = re.sub(r"<\|[^>]+\|>", " ", t)
        t = re.sub(r"</?(?:think|answer|analysis|reasoning|final_answer)>", " ", t, flags=re.IGNORECASE)
        return t.strip()

    @staticmethod
    def _normalize_num_str(s: str):
        if s is None:
            return None
        x = str(s).strip()
        if not x:
            return None
        x = x.replace("$", "").replace(",", "").replace("%", "")
        x = re.sub(r"\s+", "", x)
        x = re.sub(r"[^\d.\-+]+$", "", x).strip()
        if x in {"", "+", "-", ".", "+.", "-."}:
            return None
        return x

    @staticmethod
    def _strip_outer_brackets(s: str):
        if s is None:
            return None
        x = str(s).strip()
        pairs = {"(": ")", "[": "]", "{": "}"}
        changed = True
        while changed and len(x) >= 2:
            changed = False
            head = x[0]
            tail = x[-1]
            if head in pairs and pairs[head] == tail:
                depth = 0
                ok = True
                for i, ch in enumerate(x):
                    if ch == head:
                        depth += 1
                    elif ch == tail:
                        depth -= 1
                        if depth == 0 and i != len(x) - 1:
                            ok = False
                            break
                if ok and depth == 0:
                    x = x[1:-1].strip()
                    changed = True
        return x

    @staticmethod
    def _normalize_math_text(s: str):
        if s is None:
            return None
        x = str(s).strip()
        if not x:
            return None

        x = x.replace("$", "")
        x = x.replace("\\left", "").replace("\\right", "")
        x = x.replace("\\!", "")
        x = x.replace("\\,", "")
        x = x.replace("\\dfrac", "\\frac")
        x = x.replace("\\tfrac", "\\frac")
        x = re.sub(r"\\text\{([^{}]+)\}", r"\1", x)
        x = x.replace("\\cdot", "*")
        x = x.replace("\\times", "*")
        x = re.sub(r"[;.,]+$", "", x)
        x = re.sub(r"\s+", "", x)

        x = x.strip()
        return x if x else None

    @staticmethod
    def _latex_frac_to_plain(s: str):
        if not s:
            return s
        out = s
        while True:
            new_out = Evaluator._FRAC_RE.sub(r"(\1)/(\2)", out)
            if new_out == out:
                return out
            out = new_out

    @staticmethod
    def _extract_numeric_from_segment(segment: str, dataset: str):
        nums = [Evaluator._normalize_num_str(x) for x in Evaluator._NUM_RE.findall(segment or "")]
        nums = [x for x in nums if x is not None]
        if not nums:
            return None

        d = Evaluator._dataset_alias(dataset)
        if d == "aime2024":
            valid = []
            for x in nums:
                if not re.fullmatch(r"[+-]?\d+", x):
                    continue
                try:
                    val = int(x)
                except Exception:
                    continue
                if 0 <= abs(val) <= 999:
                    valid.append(str(val))
            if valid:
                return valid[-1]
            return None

        if d == "amc23":
            valid = [x for x in nums if re.fullmatch(r"[+-]?\d+", x)]
            return valid[-1] if valid else None

        return nums[-1]

    @staticmethod
    def _extract_expression_from_segment(segment: str):
        if not segment:
            return None
        line = segment.strip().splitlines()[0].strip()
        line = re.sub(r"^\s*[:：\-]\s*", "", line)
        return Evaluator._normalize_math_text(line)

    @staticmethod
    def _find_boxed_contents(text: str):
        results = []
        if not text:
            return results

        needle = "\\boxed{"
        i = 0
        n = len(text)
        while i < n:
            j = text.find(needle, i)
            if j < 0:
                break

            k = j + len(needle)
            depth = 1
            while k < n and depth > 0:
                ch = text[k]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                k += 1

            if depth == 0:
                full = text[j:k]
                content = text[j + len(needle) : k - 1]
                results.append((content, full, j, True))
                i = k
            else:
                # Unclosed boxed is intentionally marked as non-strict.
                full = text[j:]
                content = text[j + len(needle) :]
                results.append((content, full, j, False))
                break

        return results

    @staticmethod
    def _tail_lines_with_pos(text: str, n_lines: int = 8):
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []
        tail = lines[-n_lines:]
        block = "\n".join(tail)
        start = max(0, len(text) - len(block))

        out = []
        cursor = 0
        for ln in tail:
            out.append((ln.strip(), start + cursor))
            cursor += len(ln) + 1
        return out

    @staticmethod
    def _collect_strong_final_line_numeric(text: str, dataset: str):
        candidates = []
        for line, pos in Evaluator._tail_lines_with_pos(text, n_lines=10):
            if len(line) > 120:
                continue

            m = Evaluator._STRONG_FINAL_RE.match(line) or Evaluator._STRONG_FINAL_SHORT_RE.match(line)
            if m:
                ans = Evaluator._extract_numeric_from_segment(m.group(1), dataset)
                if ans is not None:
                    candidates.append(
                        {
                            "answer": ans,
                            "source_type": "strong_final_line",
                            "confidence_level": "strict",
                            "matched_text": line,
                            "pos": pos,
                        }
                    )
                continue

            inline_m = Evaluator._STRONG_FINAL_INLINE_RE.search(line)
            if inline_m:
                ans = Evaluator._extract_numeric_from_segment(inline_m.group(1), dataset)
                if ans is not None:
                    candidates.append(
                        {
                            "answer": ans,
                            "source_type": "strong_final_line",
                            "confidence_level": "strict",
                            "matched_text": line,
                            "pos": pos,
                        }
                    )
                continue

            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", line):
                ans = Evaluator._normalize_num_str(line)
                if ans is not None:
                    candidates.append(
                        {
                            "answer": ans,
                            "source_type": "strong_final_line",
                            "confidence_level": "strict",
                            "matched_text": line,
                            "pos": pos,
                        }
                    )
        return candidates

    @staticmethod
    def _collect_strong_final_line_math(text: str):
        candidates = []
        for line, pos in Evaluator._tail_lines_with_pos(text, n_lines=10):
            if len(line) > 180:
                continue

            m = Evaluator._STRONG_FINAL_RE.match(line) or Evaluator._STRONG_FINAL_SHORT_RE.match(line)
            if m:
                ans = Evaluator._extract_expression_from_segment(m.group(1))
                if ans is not None:
                    candidates.append(
                        {
                            "answer": ans,
                            "source_type": "strong_final_line",
                            "confidence_level": "strict",
                            "matched_text": line,
                            "pos": pos,
                        }
                    )
                continue

            inline_m = Evaluator._STRONG_FINAL_INLINE_RE.search(line)
            if inline_m:
                ans = Evaluator._extract_expression_from_segment(inline_m.group(1))
                if ans is not None:
                    candidates.append(
                        {
                            "answer": ans,
                            "source_type": "strong_final_line",
                            "confidence_level": "strict",
                            "matched_text": line,
                            "pos": pos,
                        }
                    )
                continue

            if line and len(line) <= 80 and not re.search(r"[A-Za-z]{4,}", line):
                ans = Evaluator._extract_expression_from_segment(line)
                if ans is not None and ans not in {"####", "\\boxed"}:
                    candidates.append(
                        {
                            "answer": ans,
                            "source_type": "strong_final_line",
                            "confidence_level": "strict",
                            "matched_text": line,
                            "pos": pos,
                        }
                    )
        return candidates

    @staticmethod
    def _collect_candidates_numeric(text: str, dataset: str):
        candidates = []
        clean = Evaluator._clean_text_for_extraction(text)
        if not clean:
            return candidates

        def add(answer, source_type, matched_text, pos):
            norm = Evaluator._normalize_num_str(answer)
            if norm is None:
                return
            candidates.append(
                {
                    "answer": norm,
                    "source_type": source_type,
                    "confidence_level": "strict",
                    "matched_text": matched_text,
                    "pos": pos,
                }
            )

        for m in Evaluator._HASH_RE.finditer(clean):
            ans = Evaluator._extract_numeric_from_segment(m.group(1), dataset)
            if ans is not None:
                add(ans, "hash", m.group(0), m.start())

        for content, full, pos, is_closed in Evaluator._find_boxed_contents(clean):
            if not is_closed:
                continue
            ans = Evaluator._extract_numeric_from_segment(content, dataset)
            if ans is not None:
                add(ans, "boxed", full, pos)

        candidates.extend(Evaluator._collect_strong_final_line_numeric(clean, dataset))
        return candidates

    @staticmethod
    def _collect_candidates_math(text: str):
        candidates = []
        clean = Evaluator._clean_text_for_extraction(text)
        if not clean:
            return candidates

        def add(answer, source_type, matched_text, pos):
            norm = Evaluator._normalize_math_text(answer)
            if norm is None:
                return
            candidates.append(
                {
                    "answer": norm,
                    "source_type": source_type,
                    "confidence_level": "strict",
                    "matched_text": matched_text,
                    "pos": pos,
                }
            )

        for content, full, pos, is_closed in Evaluator._find_boxed_contents(clean):
            if not is_closed:
                continue
            ans = Evaluator._extract_expression_from_segment(content)
            if ans is not None:
                add(ans, "boxed", full, pos)

        candidates.extend(Evaluator._collect_strong_final_line_math(clean))
        return candidates

    @staticmethod
    def _canonical_key(answer: str, dataset: str):
        d = Evaluator._dataset_alias(dataset)
        if answer is None:
            return None
        if d in Evaluator._NUMERIC_DATASETS:
            return Evaluator._normalize_num_str(answer)
        return Evaluator._normalize_math_text(Evaluator._latex_frac_to_plain(str(answer)))

    @staticmethod
    def _pick_best_candidate(candidates, text: str, dataset: str):
        if not candidates:
            return {
                "answer": None,
                "confidence_level": None,
                "source_type": None,
                "matched_text": None,
                "num_candidates": 0,
                "all_candidates": [],
            }

        d = Evaluator._dataset_alias(dataset)
        text_len = max(len(text), 1)
        source_base = {
            "hash": 5.5,
            "boxed": 5.4,
            "strong_final_line": 5.0,
        }

        score_map = {}
        rep_map = {}
        count_map = {}

        for c in candidates:
            value = c.get("answer")
            key = Evaluator._canonical_key(value, d) or value
            if key is None:
                continue

            recency = 0.9 * (float(c.get("pos", 0)) / float(text_len))
            score = source_base.get(c.get("source_type"), 1.0) + recency
            if d == "aime2024" and re.fullmatch(r"\d{1,3}", str(value or "")):
                score += 0.5

            score_map[key] = score_map.get(key, 0.0) + score
            count_map[key] = count_map.get(key, 0) + 1
            if key not in rep_map or c.get("pos", -1) > rep_map[key].get("pos", -1):
                rep_map[key] = c

        for key, cnt in count_map.items():
            if cnt > 1:
                score_map[key] += 0.6 * (cnt - 1)

        best_key = max(score_map.items(), key=lambda kv: kv[1])[0]
        best = rep_map[best_key]

        return {
            "answer": best.get("answer"),
            "confidence_level": "strict",
            "source_type": best.get("source_type"),
            "matched_text": best.get("matched_text"),
            "num_candidates": len(candidates),
            "all_candidates": candidates,
        }

    @staticmethod
    def extract_answer_info(text: str, dataset: str = "gsm8k"):
        d = Evaluator._dataset_alias(dataset)
        clean = Evaluator._clean_text_for_extraction(text)
        if not clean:
            return {
                "answer": None,
                "confidence_level": None,
                "source_type": None,
                "matched_text": None,
                "num_candidates": 0,
                "all_candidates": [],
            }

        if d == "math500":
            cands = Evaluator._collect_candidates_math(clean)
        else:
            cands = Evaluator._collect_candidates_numeric(clean, d)
        return Evaluator._pick_best_candidate(cands, clean, d)

    @staticmethod
    def extract_answer(text: str, dataset: str = "gsm8k"):
        return Evaluator.extract_answer_info(text, dataset=dataset).get("answer")

    @staticmethod
    def extract_true_answer(sample: dict, dataset: str = "gsm8k"):
        d = Evaluator._dataset_alias(dataset)

        if d == "gsm8k":
            raw = str(sample.get("answer", ""))
            ans = Evaluator.extract_answer(raw, dataset=d)
            if ans is not None:
                return ans
            nums = Evaluator._NUM_RE.findall(raw)
            return Evaluator._normalize_num_str(nums[-1]) if nums else None

        if d == "svamp":
            raw = str(sample.get("Answer", sample.get("answer", ""))).strip()
            ans = Evaluator._extract_numeric_from_segment(raw, dataset=d)
            if ans is not None:
                return ans
            nums = Evaluator._NUM_RE.findall(raw)
            return Evaluator._normalize_num_str(nums[-1]) if nums else None

        if d == "asdiv":
            raw = str(sample.get("answer", "")).strip()
            ans = Evaluator._extract_numeric_from_segment(raw, dataset=d)
            if ans is not None:
                return ans
            nums = Evaluator._NUM_RE.findall(raw)
            return Evaluator._normalize_num_str(nums[-1]) if nums else None

        if d == "gsmhard":
            raw = str(sample.get("target", sample.get("answer", ""))).strip()
            ans = Evaluator._normalize_num_str(raw)
            if ans is not None:
                return ans
            nums = Evaluator._NUM_RE.findall(raw)
            return Evaluator._normalize_num_str(nums[-1]) if nums else None

        if d == "aime2024":
            raw = str(sample.get("answer", "")).strip()
            m = re.search(r"\d{1,3}", raw)
            return m.group(0) if m else None

        if d == "amc23":
            raw = str(sample.get("answer", "")).strip()
            m = re.search(r"[+-]?\d+", raw)
            return m.group(0) if m else None

        # math500
        raw_answer = str(sample.get("answer", "")).strip()
        if raw_answer:
            info = Evaluator.extract_answer_info(raw_answer, dataset="math500")
            if info.get("answer") is not None:
                return info["answer"]
            norm = Evaluator._normalize_math_text(raw_answer)
            if norm:
                return norm

        raw_solution = str(sample.get("solution", "")).strip()
        if raw_solution:
            info = Evaluator.extract_answer_info(raw_solution, dataset="math500")
            if info.get("answer") is not None:
                return info["answer"]

        return None

    @staticmethod
    def _extract_answer_with_deepseek(text: str, dataset: str = "gsm8k", question: str = None, config=None):
        if config is None or not getattr(config, "USE_DEEPSEEK_EXTRACTOR", False):
            return None

        api_key = getattr(config, "DEEPSEEK_API_KEY", "")
        if not api_key:
            return None

        try:
            from openai import OpenAI
        except Exception:
            return None

        d = Evaluator._dataset_alias(dataset)
        client = OpenAI(
            api_key=api_key,
            base_url=getattr(config, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=getattr(config, "DEEPSEEK_TIMEOUT", 30),
        )

        max_chars = int(getattr(config, "DEEPSEEK_MAX_CHARS", 2500))
        clean_text = Evaluator._clean_text_for_extraction(text)[-max_chars:]

        if d == "math500":
            answer_tip = "The answer can be a mathematical expression string."
        elif d == "aime2024":
            answer_tip = "The answer must be one integer string in [0, 999]."
        elif d == "amc23":
            answer_tip = "The answer must be one integer string."
        else:
            answer_tip = "The answer must be a bare number string."

        system_prompt = (
            "You are an answer extractor.\n"
            "Only extract the final committed answer from the provided solution text.\n"
            "Do not solve the problem and do not infer missing steps.\n"
            "If the text does not clearly commit to one final answer, return null.\n"
            "Return valid JSON only with schema: {\"answer\": string|null}.\n"
            f"{answer_tip}"
        )
        user_prompt = (
            f"Question:\n{question or ''}\n\n"
            f"Solution text:\n{clean_text}\n\n"
            "Extract the final answer only and return JSON."
        )

        try:
            resp = client.chat.completions.create(
                model=getattr(config, "DEEPSEEK_MODEL", "deepseek-chat"),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=80,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            ans = data.get("answer", None)
            if ans is None:
                return None

            if d == "math500":
                return Evaluator._normalize_math_text(ans)
            if d == "aime2024":
                m = re.search(r"\d{1,3}", str(ans))
                return m.group(0) if m else None
            if d == "amc23":
                m = re.search(r"[+-]?\d+", str(ans))
                return m.group(0) if m else None
            return Evaluator._normalize_num_str(ans)
        except Exception:
            return None

    @staticmethod
    def extract_answer_robust(text: str, dataset: str = "gsm8k", question: str = None, config=None):
        info = Evaluator.extract_answer_info(text, dataset=dataset)
        local_answer = info.get("answer")
        if local_answer is not None:
            return local_answer, info

        api_answer = Evaluator._extract_answer_with_deepseek(
            text=text,
            dataset=dataset,
            question=question,
            config=config,
        )
        if api_answer is not None:
            return api_answer, {
                "answer": api_answer,
                "confidence_level": "api",
                "source_type": "deepseek_api",
                "matched_text": None,
                "num_candidates": info.get("num_candidates", 0),
                "all_candidates": info.get("all_candidates", []),
            }

        return None, {
            "answer": None,
            "confidence_level": None,
            "source_type": None,
            "matched_text": None,
            "num_candidates": info.get("num_candidates", 0),
            "all_candidates": info.get("all_candidates", []),
        }

    @staticmethod
    def _to_float_like(s: str):
        if s is None:
            return None
        x = str(s).strip()
        if not x:
            return None
        x = Evaluator._latex_frac_to_plain(x)
        x = x.replace("(", "").replace(")", "")
        x = x.replace("{", "").replace("}", "")
        x = Evaluator._normalize_num_str(x)
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            m = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", x)
            if not m:
                return None
            den = float(m.group(2))
            if abs(den) < 1e-12:
                return None
            return float(m.group(1)) / den

    @staticmethod
    def numeric_equal(a, b, atol=1e-6):
        if a is None or b is None:
            return False
        aa = Evaluator._to_float_like(a)
        bb = Evaluator._to_float_like(b)
        if aa is None or bb is None:
            return False
        return abs(aa - bb) <= atol

    @staticmethod
    def _split_top_level(expr: str, sep: str = ","):
        parts = []
        cur = []
        depth_paren = 0
        depth_brace = 0
        for ch in expr:
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)

            if ch == sep and depth_paren == 0 and depth_brace == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
        parts.append("".join(cur))
        return parts

    @staticmethod
    def answers_equivalent(pred, truth, dataset: str = "gsm8k"):
        d = Evaluator._dataset_alias(dataset)
        if pred is None or truth is None:
            return False

        if d in Evaluator._NUMERIC_DATASETS:
            return Evaluator.numeric_equal(pred, truth)

        if Evaluator.numeric_equal(pred, truth):
            return True

        p = Evaluator._normalize_math_text(Evaluator._latex_frac_to_plain(str(pred)))
        t = Evaluator._normalize_math_text(Evaluator._latex_frac_to_plain(str(truth)))
        if p is None or t is None:
            return False

        if p == t:
            return True

        if p.lower() == t.lower():
            return True

        p_items = [x.strip() for x in Evaluator._split_top_level(Evaluator._strip_outer_brackets(p), sep=",")]
        t_items = [x.strip() for x in Evaluator._split_top_level(Evaluator._strip_outer_brackets(t), sep=",")]
        if len(p_items) > 1 and len(p_items) == len(t_items):
            ok = True
            for x, y in zip(p_items, t_items):
                if x == y:
                    continue
                if Evaluator.numeric_equal(x, y):
                    continue
                if x.lower() == y.lower():
                    continue
                ok = False
                break
            if ok:
                return True

        return False

    @staticmethod
    def is_correct(pred, truth, dataset: str = "gsm8k"):
        return Evaluator.answers_equivalent(pred, truth, dataset=dataset)

    @staticmethod
    def extract_answer_info_gsm8k(text: str):
        return Evaluator.extract_answer_info(text, dataset="gsm8k")

    @staticmethod
    def extract_answer_gsm8k(text: str):
        return Evaluator.extract_answer(text, dataset="gsm8k")

    @staticmethod
    def extract_answer_robust_gsm8k(text: str, question: str = None, config=None):
        return Evaluator.extract_answer_robust(text, dataset="gsm8k", question=question, config=config)
