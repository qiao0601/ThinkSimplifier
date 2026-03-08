# -*- coding: utf-8 -*-
import json
import re


class Evaluator:
    _NUM_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")
    _HASH_RE = re.compile(r"####\s*([^\n\r]+)")
    _BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
    _BOXED_UNCLOSED_RE = re.compile(r"\\boxed\{([^}\n\r]*)")
    _FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")

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
        s = s.replace("\\cdot", "*")
        s = s.replace("\\times", "*")
        s = s.replace("，", ",")
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
                if re.fullmatch(r"[+-]?\d+", x):
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
        s = segment.strip()
        s = s.split("\n")[0].strip()
        s = re.sub(r"^\s*[:：\-]\s*", "", s)
        return Evaluator._normalize_math_text(s)

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

        for m in Evaluator._BOXED_RE.finditer(clean):
            ans = Evaluator._extract_numeric_from_segment(m.group(1), dataset)
            if ans is not None:
                add(ans, "boxed", m.group(0), m.start())

        for m in Evaluator._BOXED_UNCLOSED_RE.finditer(clean):
            ans = Evaluator._extract_numeric_from_segment(m.group(1), dataset)
            if ans is not None:
                add(ans, "boxed", m.group(0), m.start())

        strict_patterns = [
            r"(?:final\s*answer|answer)\s*[:：]?\s*([^\n\r]+)",
            r"(?:the\s*answer\s*is|thus|therefore|so)\s*[:：]?\s*([^\n\r]+)",
        ]
        for p in strict_patterns:
            for m in re.finditer(p, clean, flags=re.IGNORECASE):
                ans = Evaluator._extract_numeric_from_segment(m.group(1), dataset)
                if ans is not None:
                    add(ans, "answer_line", m.group(0), m.start())

        lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
        if lines:
            tail = lines[-1]
            if re.fullmatch(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", tail):
                add(tail, "tail_number", tail, max(0, len(clean) - len(tail)))

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

        for m in Evaluator._HASH_RE.finditer(clean):
            ans = Evaluator._extract_expression_from_segment(m.group(1))
            if ans is not None:
                add(ans, "hash", m.group(0), m.start())

        for m in Evaluator._BOXED_RE.finditer(clean):
            ans = Evaluator._extract_expression_from_segment(m.group(1))
            if ans is not None:
                add(ans, "boxed", m.group(0), m.start())

        for m in Evaluator._BOXED_UNCLOSED_RE.finditer(clean):
            ans = Evaluator._extract_expression_from_segment(m.group(1))
            if ans is not None:
                add(ans, "boxed", m.group(0), m.start())

        strict_patterns = [
            r"(?:final\s*answer|answer)\s*[:：]?\s*([^\n\r]+)",
            r"(?:the\s*answer\s*is|thus|therefore|so)\s*[:：]?\s*([^\n\r]+)",
        ]
        for p in strict_patterns:
            for m in re.finditer(p, clean, flags=re.IGNORECASE):
                ans = Evaluator._extract_expression_from_segment(m.group(1))
                if ans is not None:
                    add(ans, "answer_line", m.group(0), m.start())

        lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
        if lines:
            tail = Evaluator._extract_expression_from_segment(lines[-1])
            if tail is not None:
                add(tail, "tail_text", lines[-1], max(0, len(clean) - len(lines[-1])))

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
            "hash": 5.0,
            "boxed": 4.8,
            "answer_line": 4.3,
            "tail_number": 3.6,
            "tail_text": 3.2,
        }

        scores = {}
        meta = {}
        counts = {}

        for c in candidates:
            value = c["answer"]
            counts[value] = counts.get(value, 0) + 1

            recency = 0.9 * (c["pos"] / text_len)
            score = source_base.get(c["source_type"], 1.0) + recency

            if d == "aime2024":
                if re.fullmatch(r"\d{1,3}", value):
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
            return m.group(0) if m else Evaluator.extract_answer(raw, dataset=d)

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
            schema_tip = "{\"answer\": string|null}"
            answer_tip = "answer can be a math expression string."
        else:
            schema_tip = "{\"answer\": string|null}"
            answer_tip = "answer must be a bare number string."

        system_prompt = (
            "You are an answer extractor.\n"
            "Only extract the final committed answer from the provided solution text.\n"
            "Do not solve the problem.\n"
            "If not clearly committed, return null.\n"
            f"Return valid JSON only with schema: {schema_tip}\n"
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

        # math500
        if Evaluator.numeric_equal(pred, truth):
            return True

        p = Evaluator._normalize_math_text(Evaluator._latex_frac_to_plain(str(pred)))
        t = Evaluator._normalize_math_text(Evaluator._latex_frac_to_plain(str(truth)))
        if p is None or t is None:
            return False
        if p == t:
            return True

        # tuple/list style fallback: compare token-wise
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

    # Backward-compatible wrappers
    @staticmethod
    def extract_answer_info_gsm8k(text: str):
        return Evaluator.extract_answer_info(text, dataset="gsm8k")

    @staticmethod
    def extract_answer_gsm8k(text: str):
        return Evaluator.extract_answer(text, dataset="gsm8k")

    @staticmethod
    def extract_answer_robust_gsm8k(text: str, question: str = None, config=None):
        return Evaluator.extract_answer_robust(text, dataset="gsm8k", question=question, config=config)
