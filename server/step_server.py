import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
import torch
import torch.nn.functional as F

from utils.utils import llama_token_ids
from utils.gsm8k_dataset import STEP_INSTRUCTION, STEP_RESPONSE

model_name_or_path = os.getenv('MODEL_PATH')

tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=True)
print(f"tokenizer.pad_token_id = {tokenizer.pad_token_id }")
print("Tokenizer loaded successfully.")

step_model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
step_model.eval()
print("Step model loaded successfully.")

app = FastAPI()


class InputText(BaseModel):
    texts: List[str]


class OutputPrediction(BaseModel):
    step_reward: List[float]


@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):

    alpaca_style = []

    for qa in input_text.texts:
        alpaca_style.append(f"{STEP_INSTRUCTION}{qa.strip()}\n\n{STEP_RESPONSE}True\n\n")

    max_seq_length = 2048
    inputs = tokenizer(alpaca_style, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    print("Length of tokenized sequences:", inputs["input_ids"].shape[1])
    inputs = {name: tensor.to(step_model.device) for name, tensor in inputs.items()}
    input_ids = inputs["input_ids"]

    print(f"input ids: {input_ids}")

    outputs = step_model(**inputs)

    logits = outputs[0]  # [b, seq, voc]
    org_loss_mask = ~(input_ids == tokenizer.pad_token_id)  # [b, seq]
    xy_lengths = org_loss_mask.sum(dim=-1)

    row_index = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
    col_index = xy_lengths - 4

    print(f"logits: {logits.shape}, xy_lengths: {xy_lengths}")
    pre_tf_logits = logits[row_index, col_index, :].contiguous()  # [b, voc]

    # get True/False position
    tf_logits = pre_tf_logits[:, (llama_token_ids["True"], llama_token_ids["False"])]  # [b, 2]
    tf_probs = F.softmax(tf_logits, dim=-1)

    rewards = tf_probs[:, 0]  # True probs, [b]
    rewards = rewards.view(-1)
    rewards = rewards.detach().cpu().numpy().tolist()

    print(f"rewards: {rewards}")
    return {"step_reward": rewards}
