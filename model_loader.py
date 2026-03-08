# -*- coding: utf-8 -*-

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    def __init__(self, config):
        self.config = config
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=config.DEVICE,
            trust_remote_code=True,
        )
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.do_sample = False
        self.model.eval()
