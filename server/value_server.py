# --*-- coding: utf-8 --*--

# Copyright 2023 Tencent
"""Start MLP value set server."""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from models.value_model import ValueModel
from transformers import LlamaTokenizer

model_name_or_path = os.getenv('MODEL_PATH')

tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token_id = 2
print("Tokenizer loaded successfully.")

value_model = ValueModel.from_pretrained(model_name_or_path, device_map="auto")
value_model.eval()
print("Value model loaded successfully.")

app = FastAPI()

class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    values: List[float]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    max_seq_length = 1024
    inputs = tokenizer(input_text.texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    print("Length of tokenized sequences:", inputs["input_ids"].shape[1])
    inputs = {name: tensor.to(value_model.device) for name, tensor in inputs.items()}
    outputs = value_model(**inputs)
    values = outputs['values']
    values = values.detach().cpu().numpy().tolist()

    return {"values": values}