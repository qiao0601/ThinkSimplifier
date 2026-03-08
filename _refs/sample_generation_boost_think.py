import json
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import re # Import regex for parsing
import os
from transformers import LogitsProcessor
import numpy as np
import sys
from math_util import *

MODEL_NAME = sys.argv[1]
DATASET_LIMIT = int(sys.argv[2])
DATASET_PATH = sys.argv[3]
OUTPUT_PATH = sys.argv[4]
MAX_TOKENS = 32768
TEMPERATURE_REASONING = 0.6
TOP_P = 0.95
TOP_K = 30
presence_penalty = 1.5 if "7B" not in MODEL_NAME or "8B" not in MODEL_NAME else 1.0    

# Set tensor parallel size based on CUDA_VISIBLE_DEVICES
TENSOR_PARALLEL_SIZE = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) if os.environ.get('CUDA_VISIBLE_DEVICES') else 1
print(f"Using {TENSOR_PARALLEL_SIZE} GPUs for tensor parallelism")

PROMPT_TEMPLATE_MATH_TASK = (
    "Solve the following math problem. "
    "Directly output your final answer within \\boxed{}. DO NOT say anything else.\n\n"
    "Question: "
)

PROMPT_TEMPLATE_NQ_TASK = (
    "Answer the following question. "
    "Directly output your final answer within \\boxed{}. DO NOT say anything else.\n\n"
    "Question: "
)

model_name = MODEL_NAME
prompt_template = PROMPT_TEMPLATE_NQ_TASK if "nq" in DATASET_PATH else PROMPT_TEMPLATE_MATH_TASK

# --- Load VLLM Model ---
print(f"Loading model: {model_name}...")
if "70B" not in model_name:
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        max_model_len=MAX_TOKENS,
    )
else:
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        max_model_len=8192,
        max_num_seqs=35
    )
print("Model loaded.")


def process_token(token_ids, logits):
    boost_token_id = tokenizer("</think>", add_special_tokens=False)['input_ids'][0]
    think_token_new_line_ids = tokenizer("</think>\n", add_special_tokens=False)['input_ids']
    think_and_start_answer_ids = tokenizer("</think>\n\\", add_special_tokens=False)['input_ids']
    new_line_id = tokenizer("\n", add_special_tokens=False)['input_ids'][0]
    new_line_id_2 = tokenizer("\n\n", add_special_tokens=False)['input_ids'][0]
    eos_token_id = tokenizer.eos_token_id
    if len(token_ids) > 0:
        # If the last token is </think>, set end_of_think_flag to True
        if token_ids[-1] == boost_token_id:
            # global end_of_think_flag
            # end_of_think_flag = True
            logits[new_line_id] = 1000000.0

        # If the last token is </answer>, set end_of_answer_flag to True
        # if list(token_ids[-(len(end_of_answer_token_id1)):]) == end_of_answer_token_id1 or list(token_ids[-(len(end_of_answer_token_id1)):]) == end_of_answer_token_id2:
        #     global end_of_answer_flag
        #     end_of_answer_flag = True
            # print("end_of_answer_flag set to True")

        # If the last token is </think>\n, enforce the next token to be <answer>
        # print(tokenizer.decode(list(token_ids[-(len(think_token_new_line_ids)):])))
        # print(list(token_ids[-(len(think_token_new_line_ids)):]))
        # print(think_token_new_line_ids)
        # print('--'*20)
        if list(token_ids[-(len(think_token_new_line_ids)):]) == think_token_new_line_ids:
            # print("enforce the next token to be <answer>")
            logits[tokenizer("\\boxed{", add_special_tokens=False)['input_ids'][0]] = 1000000.0
        if list(token_ids[-(len(think_and_start_answer_ids)):]) == think_and_start_answer_ids:
            # print("enforce the next token to be <answer>")
            logits[tokenizer("boxed", add_special_tokens=False)['input_ids'][0]] = 1000000.0
    
    end_of_think_flag = boost_token_id in token_ids
    if not end_of_think_flag and len(token_ids) > 0:
        # compute std of logits
        # std = logits.std()
        bias = 0.6 * (logits.max() - logits.mean())
        logits[boost_token_id] += bias
    
    end_of_answer_flag = think_and_start_answer_ids in token_ids and (token_ids[-1] == new_line_id or token_ids[-1] == new_line_id_2)
    if end_of_answer_flag:
        logits[eos_token_id] = 1000000.0

    return logits


tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=TEMPERATURE_REASONING, max_tokens=MAX_TOKENS, logits_processors=[process_token], top_p=TOP_P, top_k=TOP_K, presence_penalty=presence_penalty)

questions = []
with open(DATASET_PATH, "r") as f:
    for line in f:
        questions.append(json.loads(line)["question"])

questions = questions[:DATASET_LIMIT]

print("Preparing chat inputs...")
chat_conversations = []
for q in questions:
    user_content = prompt_template + q
    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": ""}
    ]
    chat_conversations.append(conversation)

outputs = llm.chat(chat_conversations, sampling_params)

print("Processing results...")
results = []
box_pattern = re.compile(r"\\boxed\{(.+?)\}")
for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
    question = questions[i]
    generated_text = output.outputs[0].text.strip()
    answer = "Error: Boxed answer not found"

    matches = box_pattern.findall(generated_text)

    try:
        reasoning = generation.split("<think>")[1].split("</think>")[0].strip()
    except:
        reasoning = ""

    answer = my_answer_extraction(generated_text)
    
    result_data = {
        "question": question,
        "answer": answer,
        "confidence": "",
        "reasoning": reasoning,
        "generated_text": generated_text
    }
    results.append(result_data)

print(f"Writing results to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
