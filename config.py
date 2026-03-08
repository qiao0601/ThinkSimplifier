# -*- coding: utf-8 -*-

import os


class Config:
    DATASET_PRESETS = {
        "gsm8k": {
            "name": "tinyBenchmarks/tinyGSM8k",
            "split": "test",
            "question_key": "question",
            "answer_key": "answer",
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
        self.ANSWER_KEY = ""
        self.apply_dataset(self.DATASET_ALIAS)

        self.OUTPUT_DIR = "./results"
        self.SAMPLE_LIMIT = None

        # Strategy groups
        self.RUN_INPUT_BASED = True
        self.RUN_OUTPUT_BASED = True

        # Optional API fallback extractor
        self.USE_DEEPSEEK_EXTRACTOR = True
        self.DEEPSEEK_API_KEY = "sk-1909df3b22da49a9a466e2a3bc8ed546"
        self.DEEPSEEK_BASE_URL = "https://api.deepseek.com"
        self.DEEPSEEK_MODEL = "deepseek-chat"
        self.DEEPSEEK_TIMEOUT = 30
        self.DEEPSEEK_MAX_CHARS = 2500

        # Answer Consistency
        self.AC_K = 3
        self.AC_WINDOW_SIZE = 5
        self.AC_MIN_VALID_PROBES = 3
        self.AC_MIN_CONSECUTIVE = 2
        self.AC_PROBE_MAX_NEW_TOKENS = 20
        self.AC_MIN_TOKENS_BETWEEN_PROBES = 24
        self.AC_BOUNDARY_CHARS = ".!?\n"
        self.AC_PROBE_TAIL_CHARS = 512
        self.AC_DEBUG = False

        # ES-CoT
        self.ES_DMIN = 2.0
        self.ES_P = 0.05
        self.ES_MIN_RUNS = 3
        self.ES_MIN_STEP_ANS = 4
        self.ES_MIN_PREV_DIFFS = 2
        self.ES_STABLE_RUN_STOP = 4
        self.ES_MIN_TOKENS_BETWEEN_PROBES = 20
        self.ES_BOUNDARY_CHARS = "\n"
        self.ES_DEBUG = False

        # Confidence stop (boost-think)
        self.TTA_ALPHA = 0.9
        self.TTA_MAX_BOOST = 6.0
        self.TTA_BOUNDARY_CHARS = ".!?\n"
        self.TTA_MIN_TOKENS_BEFORE_BOOST = 80
        self.TTA_BOOST_INTERVAL = 8
        self.TTA_COMPLETION_MAX_NEW_TOKENS = 48
        self.TTA_DEBUG = False

        # Dynamic CoT
        self.DCOT_STAGE1_MAX_NEW_TOKENS = 96
        self.DCOT_STAGE2_MAX_NEW_TOKENS = 192
        self.DCOT_STAGE3_MAX_NEW_TOKENS = 384
        self.DCOT_STAGE1_BASE_THRESHOLD = 0.62
        self.DCOT_STAGE2_BASE_THRESHOLD = 0.72
        self.DCOT_COMPLEXITY_ALPHA = 0.18
        self.DCOT_DEBUG = False

    def apply_dataset(self, alias: str, split_override: str = None):
        if alias not in self.DATASET_PRESETS:
            raise ValueError(f"Unknown dataset alias: {alias}")
        preset = self.DATASET_PRESETS[alias]
        self.DATASET_ALIAS = alias
        self.DATASET_NAME = preset["name"]
        self.SPLIT = split_override if split_override else preset["split"]
        self.QUESTION_KEY = preset["question_key"]
        self.ANSWER_KEY = preset["answer_key"]
