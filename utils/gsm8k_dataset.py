# --*-- coding: utf-8 --*--

# Copyright 2023 Tencent

import copy
import json
import torch
import re
import random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from sympy import solve, sympify

INVALID_ANS = "[invalid]"
ANS_RE = re.compile(r"#### \$?(\-?[0-9\.,]+)")
STEP_INSTRUCTION = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "You are given a math problem, followed by a step-by-step reasoning process. Your task is to read the problem carefully, undstand the solving steps, and check the correctness of the last reasoning step. "
    "Output 'True' if the last step is correct, and 'False' otherwise.\n\n"
    "### Input:\n")
STEP_RESPONSE = "### Response:\n"

VALUE_INSTRUCTION = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "You are given a math problem, followed by a step-by-step reasoning process. Your task is to read the problem carefully, "
    "understand and follow the solving steps, then predict the chance of generating correct final answer with your internal knowledge. "
    "Output 'True' if you think the final answer will be correct, and 'False' otherwise.\n\n"
    "### Input:\n")
VALUE_RESPONSE = "### Response:\n"

OUTCOME_INSTRUCTION = (
    "Assess a solution including final answer to a given math problem by following below steps.\n"
    "- Evaluate the method used for solving the problem.\n"
    "- Review each calculation step for accuracy. Check for computational errors, incorrect formula applications, or arithmetic mistakes.\n"
    "- The solution should use all the information provided in the question.\n"
    "- Examine the final answer for correctness, considering the calculations and method used.\n"
    "- The final answer is seperated by '####'.\n\n"
    "### Input:\n")
OUTCOME_RESPONSE = "### Response:\nThe solution to the problem is"  # NOTE: DO NOT ADD space in the end!

STEP_CAL_INSTRUCTION = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "You are given a math problem, followed by a step-by-step reasoning process, and each step may be followed by evaluation results from sympy solver in format of [\"sympy.solve()\"] or [\"eval()\"], for example: \"This is 12 pages * 52 weeks per year = 624 pages each year. [\"eval('12*52')->624.00\"]\". Your task is to read the problem carefully, understand the solving steps, and check the correctness of the last reasoning considering both the reasoning quality and evaluation results from sympy solver, if any. "
    "Output 'True' if the last step is correct, and 'False' otherwise.\n\n"
    "### Input:\n")

def load_data(file):
    with open(file, "r", encoding='utf-8') as fin:
        data = []
        for line in fin:
            inst = json.loads(line.strip())
            data.append(inst)
    return data


def load_step_data(data, n_samples=10, step_delimiter="\n"):
    """
    Load step labeled data
    """
    org_data = []
    for inst in data:

        problem = inst["problem"]

        if "chosen_steps" in inst:
            # this is prefix data
            chosen_steps = inst["chosen_steps"]
            answer = step_delimiter.join(chosen_steps)
            step = len(chosen_steps)
            chosen = {
                'state': f"{STEP_INSTRUCTION}Question: {problem}\nAnswer: {answer}\n\n{STEP_RESPONSE}True\n\n",
                'label': 1,
                'step': step,
                'total_steps': 10,
            }
            org_data.append(chosen)

            rejected_steps = inst["rejected_steps"]
            answer = step_delimiter.join(rejected_steps)
            rejected = {
                'state': f"{STEP_INSTRUCTION}Question: {problem}\nAnswer: {answer}\n\n{STEP_RESPONSE}False\n\n",
                'label': -1,
                'step': step,
                'total_steps': 10,
            }
            org_data.append(rejected)

        else:
            # this is prm data
            generated_steps = inst["generated_steps"]
            step_labels = inst["step_labels"]

            for example, label in zip(generated_steps, step_labels):
                assert len(example) == len(label), f"Size mismatch: {len(example)} != {len(label)}"
                this_org_data = []
                for idx in range(len(example)):
                    answer = step_delimiter.join(example[:idx + 1])
                    one_label = label[idx]
                    predict_word = "True" if one_label > -1 else "False"
                    # NOTE: we always append "\n" after answer
                    this_org_data.append({
                        'state':
                            f"{STEP_INSTRUCTION}Question: {problem}\nAnswer: {answer}\n\n{STEP_RESPONSE}{predict_word}\n\n",
                        'label':
                            one_label,
                        'step':
                            idx + 1,
                        'total_steps':
                            len(example),
                    })
                # sample n_samples state from each question
                sampled_data = random.sample(this_org_data, min(n_samples, len(this_org_data)))
                org_data.extend(sampled_data)

    return org_data


def convert_to_left_equation(text):
    if "=" in text:
        left = text[:text.find("=")]
        right = text[text.find("=")+1:]
        text = left + "-" + "(" + right + ")"

    return text


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

class GSM8KDataset(Dataset):

    def __init__(self,
                 dataset_path,
                 tokenizer,
                 partition="train",
                 max_words=1024,
                 local_rank=0,
                 n_samples=2,
                 task_name="value_net",
                 train_lm=False,
                 use_calculator=False):

        self.train_lm = train_lm
        self.task_name = task_name
        self.use_calculator = use_calculator
        data = load_data(dataset_path)
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_words = max_words
        self.local_rank = local_rank
        if task_name == "step_reward":
            self.ann = load_step_data(data, n_samples=n_samples, step_delimiter="\n")
        else:
            self.ann = self.org_data(data, partition)

        if self.local_rank == 0:
            print(f"dataset_path {dataset_path}")
            print(f"Loaded {len(self.ann)} examples from {partition} split")

    def org_data(self, data, partition):
        org_data = []
        all_label = []
        all_label_by_question = []
        for inst in data:
            prompt = inst['prompt']
            prompt = VALUE_INSTRUCTION + prompt.split('\n\n')[-1] + " "  # remove few-shot prompting and append a space
            # prompt is in format of "Question: xxxx\nAnswer: "
            correct_answer = self.get_reward(inst['label'], is_print=True)
            if partition == "train":
                inst['generated_text'].append(inst['label'])
            n_generate_samples = len(inst['generated_text'])

            is_have_correct_answer = False
            for i in range(n_generate_samples):
                generated_text = inst['generated_text'][i].strip()
                if len(generated_text) == 0:
                    continue
                candidate_answer = self.get_reward(generated_text, is_print=False)
                if candidate_answer == correct_answer:
                    # all the correct_answer are leagl, so no need to check the INVALID_ANS here
                    label = 1
                    is_have_correct_answer = True
                else:
                    try:
                        if eval(candidate_answer) == eval(correct_answer):
                            label = 1
                            is_have_correct_answer = True
                        else:
                            label = -1
                    except:
                        label = -1
                if label == 1:
                    predict_word = "True" if self.train_lm else ""
                else:
                    predict_word = "False" if self.train_lm else ""

                all_label.append(label)

                this_org_data = []
                generated_text_step = generated_text.split('\n')
                n_steps = len(generated_text_step)
                this_prompt = copy.deepcopy(prompt)

                for j in range(n_steps):
                    if self.use_calculator:
                        generated_text_step[j] = self.add_calculator(generated_text_step[j])
                    if j == n_steps - 1:
                        this_prompt += generated_text_step[j]
                    else:
                        this_prompt += generated_text_step[j] + '\n'

                    if self.train_lm:
                        # LM task
                        pred_str = f"{VALUE_INSTRUCTION}{this_prompt.strip()}\n\n{VALUE_RESPONSE}{predict_word}\n\n"
                    else:
                        pred_str = copy.deepcopy(this_prompt)
                    this_org_data.append({
                        'state': pred_str,
                        'label': label,
                        'step': j + 1,
                        'total_steps': n_steps + 1,
                    })

                # sample n_samples state from each question
                sampled_data = random.sample(this_org_data, min(self.n_samples, len(this_org_data)))
                org_data.extend(sampled_data)
            if is_have_correct_answer:
                all_label_by_question.append(1)
            else:
                all_label_by_question.append(0)
        if self.local_rank == 0:
            print(
                f'Mean reward {np.mean(all_label):2f}, {np.mean(all_label_by_question):2f} of questions have at least one correct answer'
            )
        return org_data

    def get_reward(self, completion, is_print=False):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            match_str = match_str.replace("$", "")
            return match_str
        else:
            if is_print:
                print(f'Answer not found: {completion}')
            return INVALID_ANS

    @staticmethod
    def add_calculator(step):
        p3 = re.compile('<<[^<>]*>>')
        p4 = re.compile('<<([^=<>]*)=([^=<>]*)>>')
        raw_expr_list = re.findall(p3, step)

        for opseq_text in raw_expr_list:
            filtered_opseq_text = opseq_text.replace("$", "").replace(",", "").replace("[", "(").replace("]", ")").replace("^", "**")
            m = p4.match(filtered_opseq_text)
            if m:
                v0, v1 = m.group(1, 2)
                if "%" in v0 and "+" not in v0 and "/" not in v0:
                    v0 = v0.replace("%", "*0.01")
                try:
                    # without any unknowns
                    step = step[:step.find(opseq_text)+len(opseq_text)] + "<calc>" + str(eval(v0))+"</calc>" + step[step.find(opseq_text)+len(opseq_text):]
                except:
                    try:
                        expression = convert_to_left_equation(filtered_opseq_text.replace("<<","").replace(">>",""))
                        variables = sympify(expression).free_symbols
                        result = solve(expression)
                        # currently only consider one unknown
                        if len(result) == 1 and len(variables) == 1:
                            variable = str(variables.pop())
                            # to filter unit word ("pill", "hour")
                            if len(variable) == 1:
                                short_result = str(result[-1])
                                if is_float(result[-1]):
                                    short_result = str(eval(str(result[-1])))
                                step = (step[:step.find(opseq_text) + len(opseq_text)] + "<calc>" + variable + "=" +
                                        short_result + "</calc>" + step[step.find(opseq_text) + len(opseq_text):])
                    except:
                        # Replace instances of '2x' with '2*x' using regular expressions
                        # opseq_text_new = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', opseq_text)
                        continue

        return step


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        example_tokens = ann['state']
        label = ann['label']
        step = ann['step']
        total_steps = ann['total_steps']

        example_ids = self.tokenizer.encode(example_tokens)
        example_ids.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example_ids, dtype=torch.int64)

        example_mask = example.ge(-1)
        example[~example_mask] = 0
        example_mask = example_mask.float()

        pad_length = self.max_words - example.shape[0]
        if pad_length > 0:
            example = F.pad(example, pad=(0, pad_length), mode='constant', value=self.tokenizer.pad_token_id)
            example_mask = F.pad(example_mask, pad=(0, pad_length), mode='constant', value=0)
        else:
            example = example[:self.max_words]
            example_mask = example_mask[:self.max_words]

        print(f"example_tokens: {example_tokens}, example_ids: {example_ids}, example: {example.shape}")

        return {
            "input_ids": example,
            "labels": label,
            "attention_mask": example_mask,
            "step": step,
            "total_steps": total_steps,
        }
