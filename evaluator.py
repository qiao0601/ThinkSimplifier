# -*- coding: utf-8 -*-
import json
import re


class Evaluator:
    _NUM_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")
    _HASH_RE = re.compile(r"####\s*([^\n\r]+)")
    _BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
    _BOXED_UNCLOSED_RE = re.compile(r"\\boxed\{([^}\n\r]*)")
    _FRAC_RE = re.compile(r"\\(?:frac|dfrac|tfrac)\{([^{}]+)\}\{([^{}]+)\}")
    _STRONG_FINAL_RE = re.compile(
        r"^\s*(?:the\s+)?final\s+answer(?:\s+is)?\s*[:：]?\s*([^\n\r]+)\s*$",
        flags=re.IGNORECASE,
    )
    _STRONG_FINAL_SHORT_RE = re.compile(
        r"^\s*final\s*[:：]\s*([^\n\r]+)\s*$",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def _dataset_alias(dataset: str):
        d = (dataset or "gsm8k").lower()
        if d in {"aime", "aime2024"}:
            return "aime2024"
        if d in {"math", "math500"}:
            return "math500"
        return "gsm8k"

    @staticmethod
    def _clean_text_for_extraction(text: str):
        if not text:
            return ""
        text = re.sub(r"<\|[^>]+\|>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        return text.strip()

    @staticmethod
    def _normalize_num_str(s: str):
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        s = s.replace("$", "").replace(",", "").replace("%", "")
        s = s.replace("：", ":").replace("，", ",")
        s = re.sub(r"[^\d\.\-\+]+$", "", s).strip()
        if s in {"", "+", "-", ".", "+.", "-."}:
            return None
        return s

    @staticmethod
    def _normalize_math_text(s: str):
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        s = s.replace("$", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\!", "")
        
        # unify common LaTeX fraction variants
        s = s.replace("\\dfrac", "\\frac")
        s = s.replace("\\tfrac", "\\frac")
        #\text{Evelyn} Evelyn
        s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)

        s = s.replace("\\cdot", "*")
        s = s.replace("\\times", "*")
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[;.,]+$", "", s)
        return s if s else None

    @staticmethod
    def _latex_frac_to_plain(s: str):
        if not s:
            return s
        while True:
            new_s = Evaluator._FRAC_RE.sub(r"(\1)/(\2)", s)
            if new_s == s:
                return s
            s = new_s

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
                    v = int(x)
                except Exception:
                    continue
                if 0 <= abs(v) <= 999:
                    valid.append(str(v))
            if valid:
                return valid[-1]
        return nums[-1]

    @staticmethod
    def _extract_expression_from_segment(segment: str):
        if not segment:
            return None
        s = segment.strip().split("\n")[0].strip()
        s = re.sub(r"^\s*[:：\-]\s*", "", s)
        return Evaluator._normalize_math_text(s)

    @staticmethod
    def _find_boxed_contents(text: str):
        """
        Extract \\boxed{...} with brace matching.
        Handles nested braces, e.g. \\boxed{(3,\\frac{\\pi}{2})}.
        Returns list[(content, full_match, start_pos)].
        """
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
                results.append((content, full, j))
                i = k
            else:
                full = text[j:]
                content = text[j + len(needle) :]
                results.append((content, full, j))
                break
        return results

    @staticmethod
    def _collect_strong_final_line_numeric(text: str, dataset: str):
        candidates = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return candidates

        tail_lines = lines[-8:]
        tail_block = "\n".join(tail_lines)
        tail_start = max(0, len(text) - len(tail_block))
        cursor = 0
        for line in tail_lines:
            if len(line) > 120:
                cursor += len(line) + 1
                continue
            m = Evaluator._STRONG_FINAL_RE.match(line) or Evaluator._STRONG_FINAL_SHORT_RE.match(line)
            if not m:
                cursor += len(line) + 1
                continue
            ans = Evaluator._extract_numeric_from_segment(m.group(1), dataset)
            if ans is None:
                cursor += len(line) + 1
                continue
            candidates.append(
                {
                    "answer": ans,
                    "source_type": "strong_final_line",
                    "confidence_level": "strict",
                    "matched_text": line,
                    "pos": tail_start + cursor,
                }
            )
            cursor += len(line) + 1
        return candidates

    @staticmethod
    def _collect_strong_final_line_math(text: str):
        candidates = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return candidates

        tail_lines = lines[-8:]
        tail_block = "\n".join(tail_lines)
        tail_start = max(0, len(text) - len(tail_block))
        cursor = 0
        for line in tail_lines:
            if len(line) > 160:
                cursor += len(line) + 1
                continue
            m = Evaluator._STRONG_FINAL_RE.match(line) or Evaluator._STRONG_FINAL_SHORT_RE.match(line)
            if not m:
                cursor += len(line) + 1
                continue
            ans = Evaluator._extract_expression_from_segment(m.group(1))
            if ans is None:
                cursor += len(line) + 1
                continue
            candidates.append(
                {
                    "answer": ans,
                    "source_type": "strong_final_line",
                    "confidence_level": "strict",
                    "matched_text": line,
                    "pos": tail_start + cursor,
                }
            )
            cursor += len(line) + 1
        return candidates

    @staticmethod
    def _collect_candidates_numeric(text: str, dataset: str):
        candidates = []
        clean = Evaluator._clean_text_for_extraction(text)
        if not clean:
            return candidates

        def add(answer, source_type, matched_text, pos):
            answer = Evaluator._normalize_num_str(answer)
            if answer is None:
                return
            candidates.append(
                {
                    "answer": answer,
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

        for content, full, pos in Evaluator._find_boxed_contents(clean):
            ans = Evaluator._extract_numeric_from_segment(content, dataset)
            if ans is not None:
                add(ans, "boxed", full, pos)

        for c in Evaluator._collect_strong_final_line_numeric(clean, dataset):
            candidates.append(c)

        return candidates

    @staticmethod
    def _collect_candidates_math(text: str):
        candidates = []
        clean = Evaluator._clean_text_for_extraction(text)
        if not clean:
            return candidates

        def add(answer, source_type, matched_text, pos):
            answer = Evaluator._normalize_math_text(answer)
            if answer is None:
                return
            candidates.append(
                {
                    "answer": answer,
                    "source_type": source_type,
                    "confidence_level": "strict",
                    "matched_text": matched_text,
                    "pos": pos,
                }
            )

        # For math500, do not use hash candidates.
        for content, full, pos in Evaluator._find_boxed_contents(clean):
            ans = Evaluator._extract_expression_from_segment(content)
            if ans is not None:
                add(ans, "boxed", full, pos)

        for c in Evaluator._collect_strong_final_line_math(clean):
            candidates.append(c)

        return candidates

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
            "hash": 5.2,
            "boxed": 5.0,
            "strong_final_line": 4.4,
        }

        scores = {}
        meta = {}
        counts = {}
        for c in candidates:
            value = c["answer"]
            counts[value] = counts.get(value, 0) + 1
            recency = 0.9 * (c["pos"] / text_len)
            score = source_base.get(c["source_type"], 1.0) + recency
            if d == "aime2024" and re.fullmatch(r"\d{1,3}", value):
                score += 0.5
            scores[value] = scores.get(value, 0.0) + score
            if value not in meta or c["pos"] > meta[value]["pos"]:
                meta[value] = c

        for value, cnt in counts.items():
            if cnt > 1:
                scores[value] += 0.6 * (cnt - 1)

        best_value = max(scores.items(), key=lambda x: x[1])[0]
        best_meta = meta[best_value]
        return {
            "answer": best_value,
            "confidence_level": "strict",
            "source_type": best_meta["source_type"],
            "matched_text": best_meta["matched_text"],
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
            candidates = Evaluator._collect_candidates_math(clean)
        else:
            candidates = Evaluator._collect_candidates_numeric(clean, d)
        return Evaluator._pick_best_candidate(candidates, clean, d)

    @staticmethod
    def extract_answer(text: str, dataset: str = "gsm8k"):
        return Evaluator.extract_answer_info(text, dataset=dataset)["answer"]

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

        if d == "math500":
            raw = str(sample.get("answer", "")).strip()
            if not raw:
                raw = str(sample.get("solution", "")).strip()
            return Evaluator._normalize_math_text(raw)

        if d == "aime2024":
            raw = str(sample.get("answer", "")).strip()
            m = re.search(r"\d{1,3}", raw)
            return m.group(0) if m else None

        raw = str(sample.get("answer", "")).strip()
        return raw if raw else None

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

        max_chars = getattr(config, "DEEPSEEK_MAX_CHARS", 2500)
        clean_text = Evaluator._clean_text_for_extraction(text)[-max_chars:]

        if d == "math500":
            answer_tip = "The answer can be a mathematical expression string."
        elif d == "aime2024":
            answer_tip = "The answer must be one integer string in [0,999]."
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
                max_tokens=64,
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
            return Evaluator._normalize_num_str(ans)
        except Exception:
            return None

    @staticmethod
    def extract_answer_robust(text: str, dataset: str = "gsm8k", question: str = None, config=None):
        info = Evaluator.extract_answer_info(text, dataset=dataset)
        local_answer = info["answer"]
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
        s = str(s).strip()
        s = Evaluator._latex_frac_to_plain(s)
        s = s.replace("(", "").replace(")", "")
        s = s.replace("{", "").replace("}", "")
        s = Evaluator._normalize_num_str(s)
        if s is None:
            return None
        try:
            return float(s)
        except Exception:
            m = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", s)
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
    def answers_equivalent(pred, truth, dataset: str = "gsm8k"):
        d = Evaluator._dataset_alias(dataset)
        if pred is None or truth is None:
            return False

        if d in {"gsm8k", "aime2024"}:
            return Evaluator.numeric_equal(pred, truth)

        if Evaluator.numeric_equal(pred, truth):
            return True

        p = Evaluator._normalize_math_text(Evaluator._latex_frac_to_plain(str(pred)))
        t = Evaluator._normalize_math_text(Evaluator._latex_frac_to_plain(str(truth)))
        if p is None or t is None:
            return False
        if p == t:
            return True

        if "," in p and "," in t:
            p_items = [x for x in p.split(",") if x != ""]
            t_items = [x for x in t.split(",") if x != ""]
            if len(p_items) == len(t_items):
                ok = True
                for x, y in zip(p_items, t_items):
                    if not (x == y or Evaluator.numeric_equal(x, y)):
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
