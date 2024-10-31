# --*-- coding: utf-8 --*--

# Copyright 2023 Tencent
"""Start RMGPT 70b server."""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch

from models.rmgpt_model import RMGPTModel
from tokenizer.tokenizer import build_tokenizer

model_name_or_path = os.getenv('MODEL_PATH')
vocab_file = os.getenv('VOCAB_FILE')
activation = os.getenv('ACTIVATION')


class tokenizer_args():

    def __init__(self, vocab_file=None, added_tokens_file=None) -> None:
        self.rank = 0
        self.vocab_file = vocab_file
        self.added_tokens_file = added_tokens_file
        self.tokenizer_type = "LLaMATokenizerThreee"
        self.pad_vocab_size_to = 128256
        self.make_vocab_size_divisible_by = None


args = tokenizer_args(vocab_file=vocab_file)
tokenizer = build_tokenizer(args)
pad_id = tokenizer.pad
print(f"Tokenizer loaded successfully, pad id: {pad_id}")

# value_model = RMGPTModel.from_pretrained(model_name_or_path, device_map="auto")
value_model = RMGPTModel.from_pretrained(model_name_or_path, device_map="auto").to(dtype=torch.bfloat16)
value_model.eval()
print(f"Value model loaded successfully.")

app = FastAPI()


class InputText(BaseModel):
    texts: List[str]


class OutputPrediction(BaseModel):
    values: List[float]


@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    max_seq_length = 2048
    example_ids = []
    for one_text in input_text.texts:
        print(f"example: {one_text}")
        one_text_ids = tokenizer.tokenize(one_text)
        if len(one_text_ids) < max_seq_length:
            one_text_ids.extend([pad_id] * (max_seq_length - len(one_text_ids)))
        else:
            one_text_ids = one_text_ids[:max_seq_length]
        example_ids.append(one_text_ids)

    example_ids = torch.IntTensor(example_ids)
    inputs = {"input_ids": example_ids}
    inputs = {name: tensor.to(value_model.device) if name == "input_ids" else tensor.to(value_model.device, dtype=torch.bfloat16) for name, tensor in inputs.items()}

    input_ids = inputs["input_ids"]
    # print(f"input_ids: {input_ids}")

    loss_mask = torch.ones(input_ids.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[input_ids == pad_id] = 0.0
    loss_mask = loss_mask.to(dtype=torch.bfloat16)
    # print(f"loss_mask: {loss_mask.shape}")

    outputs = value_model(loss_mask=loss_mask, **inputs)
    values = outputs['values']
    # print(f"raw values: {values}")
    values = values.detach().cpu().float().numpy().tolist()
    # print(f"values: {values}")

    return {"values": values}