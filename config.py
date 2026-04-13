# -*- coding: utf-8 -*-

import os


class Config:
    SYSTEM_INTEGRATION_TEST_MODES = (
        "none",
        "integration_only",
        "unit_and_integration",
        "mainline_then_integration",
    )

    DATASET_PRESETS = {
        "gsm8k": {
            "name": "tinyBenchmarks/tinyGSM8k",
            "split": "test",
            "question_key": "question",
            "answer_key": "answer",
        },
        "svamp": {
            "name": "ChilleD/SVAMP",
            "split": "test",
            "question_key": "question_concat",
            "answer_key": "Answer",
        },
        "asdiv": {
            "name": "EleutherAI/asdiv",
            "split": "validation",
            "question_keys": ["body", "question"],
            "question_joiner": "\n",
            "answer_key": "answer",
        },
        "gsmhard": {
            "name": "reasoning-machines/gsm-hard",
            "split": "train",
            "question_key": "input",
            "answer_key": "target",
        },
        "math500": {
            "name": "HuggingFaceH4/MATH-500",
            "split": "test",
            "question_key": "problem",
            "answer_key": "answer",
        },
        "aime2024": {
            "name": "HuggingFaceH4/aime_2024",
            "split": "train",
            "question_key": "problem",
            "answer_key": "answer",
        },
        "amc23": {
            "name": "math-ai/amc23",
            "split": "test",
            "question_key": "question",
            "answer_key": "answer",
        },
    }

    DATASET_GROUPS = {
        "enhance": ["gsm8k", "svamp", "asdiv"],
        "validate": ["aime2024", "amc23", "gsmhard"],
        "mainline": ["gsm8k", "svamp", "asdiv", "aime2024", "amc23", "gsmhard"],
    }

    def __init__(self):
        # Model
        self.MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        self.MAX_NEW_TOKENS = 1024
        self.DEVICE = "auto"

        # Dataset (can be overwritten by CLI)
        self.DATASET_ALIAS = "gsm8k"
        self.DATASET_NAME = ""
        self.SPLIT = ""
        self.QUESTION_KEY = ""
        self.QUESTION_KEYS = []
        self.QUESTION_JOINER = "\n"
        self.ANSWER_KEY = ""
        self.apply_dataset(self.DATASET_ALIAS)

        self.OUTPUT_DIR = "./results"
        self.SAMPLE_LIMIT = None

        # Strategy groups
        self.RUN_INPUT_BASED = True
        self.RUN_OUTPUT_BASED = True

        # Integration mode:
        # none: run unit only
        # integration_only: run integration only
        # unit_and_integration: run unit then integration on selected datasets
        # mainline_then_integration: force mainline unit, then integration
        self.SYSTEM_INTEGRATION_TEST_MODE = "none"

        # Optional API fallback extractor
        self.USE_DEEPSEEK_EXTRACTOR = True
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_BASE_URL = "https://api.deepseek.com"
        self.DEEPSEEK_MODEL = "deepseek-chat"
        self.DEEPSEEK_TIMEOUT = 30
        self.DEEPSEEK_MAX_CHARS = 2500

        # Mainline single tuned set:
        # Enhance: gsm8k/svamp/asdiv; Validate: aime2024/amc23/gsmhard
        # Conservative defaults to reduce early-wrong stopping.

        # Answer Consistency (Answer Convergence inspired)
        self.AC_K = 4
        self.AC_WINDOW_SIZE = 8
        self.AC_MIN_VALID_PROBES = 4
        self.AC_MIN_CONSECUTIVE = 3
        self.AC_MIN_PROGRESS_RATIO = 0.55
        self.AC_REQUIRE_LAST_PROBE_STRICT = True
        self.AC_PROBE_MAX_NEW_TOKENS = 20
        self.AC_MIN_TOKENS_BETWEEN_PROBES = 40
        self.AC_BOUNDARY_CHARS = ".!\n"
        self.AC_PROBE_TAIL_CHARS = 512
        self.AC_DEBUG = False

        # ES-CoT (run-jump signal + late strong-stable fallback)
        self.ES_DMIN = 2.2
        self.ES_P = 0.01
        self.ES_MIN_RUNS = 3
        self.ES_MIN_STEP_ANS = 6
        self.ES_MIN_PREV_DIFFS = 2
        self.ES_MIN_RUN_AFTER_JUMP = 4
        self.ES_STRONG_STABLE_RUN_STOP = 6
        self.ES_MIN_PROGRESS_RATIO = 0.62
        self.ES_MIN_TOKENS_BETWEEN_PROBES = 36
        self.ES_BOUNDARY_CHARS = "\n"
        self.ES_DEBUG = False

        # Confidence stop (stability-driven gradual boost)
        self.TTA_ALPHA = 0.30
        self.TTA_MAX_BOOST = 4.5
        self.TTA_BOUNDARY_CHARS = ".!?\n"
        self.TTA_MIN_TOKENS_BEFORE_BOOST = 176
        self.TTA_BOOST_INTERVAL = 6
        self.TTA_PROBE_GAP = 48
        self.TTA_STABLE_MIN_RUN = 3
        self.TTA_MIN_PROGRESS_RATIO = 0.52
        self.TTA_COMPLETION_MAX_NEW_TOKENS = 48
        self.TTA_DEBUG = False

        # Dynamic CoT (rule-first, reward-second; less aggressive stage cuts)
        self.MAX_POSITION_EMBEDDINGS = None
        self.MODEL_MAX_LENGTH = None

        self.DCOT_STAGE1_MAX_NEW_TOKENS = 128
        self.DCOT_STAGE2_MAX_NEW_TOKENS = 320
        self.DCOT_STAGE3_MAX_NEW_TOKENS = 512
        self.DCOT_STAGE1_BASE_THRESHOLD = 0.64
        self.DCOT_STAGE2_BASE_THRESHOLD = 0.78
        self.DCOT_COMPLEXITY_ALPHA = 0.06
        self.DCOT_STAGE1_STOP_MIN_PROGRESS = 0.78
        self.DCOT_STAGE2_STOP_MIN_PROGRESS = 0.62
        self.DCOT_DEBUG = False

    def apply_dataset(self, alias: str, split_override: str = None):
        if alias not in self.DATASET_PRESETS:
            raise ValueError(f"Unknown dataset alias: {alias}")
        preset = self.DATASET_PRESETS[alias]
        self.DATASET_ALIAS = alias
        self.DATASET_NAME = preset["name"]
        self.SPLIT = split_override if split_override else preset["split"]
        self.QUESTION_KEY = preset.get("question_key", "")
        self.QUESTION_KEYS = list(preset.get("question_keys", []))
        self.QUESTION_JOINER = preset.get("question_joiner", "\n")
        self.ANSWER_KEY = preset["answer_key"]
