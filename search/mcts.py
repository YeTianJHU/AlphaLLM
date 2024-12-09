# --*-- coding: utf-8 --*--

# Copyright 2023 Tencent
"""MCTS"""

import time
import re
import json
import numpy as np
import openai
import requests
import pickle
import copy
import sys
import random
from pathlib import Path
from argparse import Namespace
from collections import defaultdict, Counter, deque

parent_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_directory))
from utils.utils import format_llama2, llama_token_ids, llama3_token_ids, WIZARD_MATH_PROMPT
from utils.gsm8k_dataset import STEP_INSTRUCTION, STEP_CAL_INSTRUCTION, STEP_RESPONSE, VALUE_INSTRUCTION, VALUE_RESPONSE, OUTCOME_INSTRUCTION, OUTCOME_RESPONSE, GSM8KDataset
from server.extract_client_example import get_cal as get_math_extractor
from utils.tools_api import code_api_remove_leftover

np.set_printoptions(3)

ANS_RE = re.compile(r"#### \$?(\-?[0-9\.,]+)")  
INVALID_ANS = "[invalid]"
STOP_TOKEN_WIZARDMATH = "</s>"
STOP_TOKEN_LLAMA3 = "<|eot_id|>"
DEBUG = False

MODEL_MERGE_INSTRUCTION = "Assess whether the two reasoning steps (A and B) utilize equivalent mathematical expressions, operations, and variables to solve the problem, considering the given context and any additional information provided. If the steps use different variables or expressions, but the underlying mathematical operations and principles are the same, answer \"yes\". Otherwise, answer \"no'. Also, provide a concise explanation for your answer."
MODEL_MERGE_TEMPLATE = '''
The following illustrates a logical deduction process (referred to as A) alongside an isolated reasoning step (referred to as B) applied to the identical mathematical problem Q:
```
Q: {question}
A: {action_0}
B: {action_1}
```

{instruction}
Only answer with `yes` or `no`.
'''

IMPROVE_LLAMA_POLICY_PROMPT = '''A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions.
User: {query}
Assistant: '''

LLAMA3_PROMPT = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful agent on solving math problems.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{QUERY}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''


def select_url(url):
    '''
    randomly select an url 
    '''
    if ',' in url:
        urls = url.split(',')
        return urls[random.randint(0, len(urls) - 1)]
    else:
        return url


class CustomCompletion(openai.api_resources.abstract.APIResource):

    @classmethod
    def create(cls, *args, **kwargs):
        url = "/completions"
        response = cls._static_request("post", url, *args, **kwargs)
        return response


def weighted_vote(pred_answer_numbers, weights):
    weighted_votes = Counter()
    for i, answer in enumerate(pred_answer_numbers):
        weighted_votes[answer] += weights[i]
    return weighted_votes.most_common(1)[0][0]


def get_step_reward_70b(step_reward_url, prompt, true_text='True', false_text='False', tokenizer_type='llama2'):
    """
    Request the vllm server for 70b model
    NOTE: We don't have end of trajctory information in training, thus,
            we train step reward without using the '\n' at the end, let's remove it first.
            and do not remove the beginning space ' '.
    """
    payload = {"prompt": prompt, "max_tokens": 1, "logprobs": 10, "temperature": 0}
    if DEBUG:
        print("get_step_reward_70b", payload)
    response = requests.post(step_reward_url, json=payload)

    if tokenizer_type == 'llama2':
        token_ids = llama_token_ids
    elif tokenizer_type == 'llama3':
        token_ids = llama3_token_ids

    if response.status_code == 200:
        result = response.json()
        logprobs = result['logprobs'][0][0]
        if DEBUG:
            print('logprobs', logprobs)
        if str(token_ids[true_text]) not in logprobs and str(token_ids[false_text]) not in logprobs:
            return 0.0
        elif str(token_ids[true_text]) not in logprobs:
            return 0.0
        elif str(token_ids[false_text]) not in logprobs:
            return 1.0
        else:
            prob_true = np.exp(logprobs[str(token_ids[true_text])])
            prob_false = np.exp(logprobs[str(token_ids[false_text])])
            return prob_true / (prob_true + prob_false)
    else:
        print("Error:", response.status_code, response.text)
        return None


def get_outcome_70b(orm_url, prompt, orm_calibrate_logits=False, tokenizer_type='llama2'):
    """
    Request the vllm server for 70b model
    NOTE: We don't have end of trajctory information in training, thus,
            we train step reward without using the '\n' at the end, let's remove it first.
            and do not remove the beginning space ' '.
            The answer2end here should be a complete answer from begining to end.
    """

    if tokenizer_type == 'llama2':
        token_ids = llama_token_ids
        payload = {"prompt": prompt, "max_tokens": 1, "logprobs": 10, "temperature": 0}
    elif tokenizer_type == 'llama3':
        token_ids = llama3_token_ids
        payload = {
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 5,
            "temperature": 0
        }  # logprobs need to be 5 for vllm for llama3

    if DEBUG:
        print("get_outcome_70b", payload)
    orm_url = select_url(orm_url)
    response = requests.post(orm_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        logprobs = result['logprobs'][0][0]
        if DEBUG:
            print('logprobs', logprobs)
        if orm_calibrate_logits:
            total_logits = sum([np.exp(vv) for vv in logprobs.values()])
            if str(token_ids['correct']) in logprobs:
                prob_true = np.exp(logprobs[str(token_ids['correct'])])
                prob = prob_true / total_logits
            elif str(token_ids['wrong']) in logprobs:
                prob_false = np.exp(logprobs[str(token_ids['wrong'])])
                prob = 1 - prob_false / total_logits
            else:
                prob = 0.0
            return prob
        else:
            if str(token_ids['correct']) not in logprobs and str(token_ids['wrong']) not in logprobs:
                return 0.0
            elif str(token_ids['correct']) not in logprobs:
                return 0.0
            elif str(token_ids['wrong']) not in logprobs:
                return 1.0
            else:
                prob_true = np.exp(logprobs[str(token_ids['correct'])])
                prob_false = np.exp(logprobs[str(token_ids['wrong'])])
                return prob_true / (prob_true + prob_false)
    else:
        print("Error:", response.status_code, response.text)
        return None

def get_value_70b(value_url, question_for_value, cur_answer):
    """
    Deprecated:
    Request the vllm server for 70b model
    NOTE: we train value net without using the '\n' at the end, let's remove it first.
            and do not remove the beginning space ' '.
    """
    cur_answer = cur_answer.rstrip()
    assert len(cur_answer) > 0, f"Cur answer must be longer than 0! This should not be root"
    alpaca_style = f"{VALUE_INSTRUCTION}{question_for_value}{cur_answer}\n\n{VALUE_RESPONSE}"
    payload = {"prompt": alpaca_style, "max_tokens": 1, "logprobs": 10, "temperature": 0}
    response = requests.post(value_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        logprobs = result['logprobs'][0][0]
        if DEBUG:
            print('value_url', value_url, 'logprobs', logprobs)
        if str(llama_token_ids['True']) not in logprobs and str(llama_token_ids['False']) not in logprobs:
            return 0.0
        elif str(llama_token_ids['True']) not in logprobs:
            return 0.0
        elif str(llama_token_ids['False']) not in logprobs:
            return 1.0
        else:
            prob_true = np.exp(logprobs[str(llama_token_ids['True'])])
            prob_false = np.exp(logprobs[str(llama_token_ids['False'])])
            return prob_true / (prob_true + prob_false)
    else:
        print("Error:", response.status_code, response.text)
        return None


def get_step_reward(step_reward_url, question_for_value, cur_answer):
    """
    MLP classification task
    NOTE: We don't have end of trajctory information in training, thus,
            we train step reward without using the '\n' at the end, let's remove it first.
            and do not remove the beginning space ' '.
    """
    cur_answer = cur_answer.rstrip()
    assert len(cur_answer) > 0, f"Cur answer must be longer than 0! This should not be root"
    payload = {"texts": [question_for_value + cur_answer]}
    if DEBUG:
        print('get_step_reward', payload)
    response = requests.post(step_reward_url, json=payload)
    if response.status_code == 200:
        result = response.json()
        if DEBUG:
            print("step_reward:", result["step_reward"])
        return result["step_reward"][0]
    else:
        if DEBUG:
            print("Values Error:", response.status_code, response.text)
        return None


def get_value(value_url, prompt):
    """
    MLP classification task.
    """

    payload = {"texts": [prompt]}
    if DEBUG:
        print('get_value', payload)

    response = requests.post(value_url, json=payload)
    if response.status_code == 200:
        result = response.json()
        if DEBUG:
            print("value_url", value_url, "Values:", result["values"])
        return result["values"][0]
    else:
        if DEBUG:
            print("Values Error:", response.status_code, response.text)
        return None


def find_equations_in_text(text):
    """
    Find if equations exist in the text.
    """
    p0 = re.compile("\$.*?\$")
    p1 = re.compile('<<[^<>]*>>')
    raw_expr_list0 = re.findall(p0, text)
    raw_expr_list1 = re.findall(p1, text)
    raw_expr_list = raw_expr_list0 + raw_expr_list1
    if raw_expr_list:
        return True
    else:
        return False


def check_code_pattern(content):
    """
    Check if the content contains both code and solution pattern.
    """
    code_pattern = r"<code>(.*?)</code>"
    solution_pattern = r"<solution>(.*?)</solution>"

    code_match = re.findall(code_pattern, content, re.DOTALL)
    solution_match = re.findall(solution_pattern, content, re.DOTALL)

    return True if code_match and solution_match else False


class ProblemState():

    def __init__(self,
                 question,
                 answer,
                 config,
                 cur_answer='',
                 cur_step=0,
                 max_lines=10,
                 is_terminate=False,
                 cur_answer_w_math_extractor="",
                 has_code_exec=False):
        """
        All service calls should be taken place inside this class.
        cur_answer: current answer (multi-steps with a space at the very beginning)
        """
        for key, value in config.items():
            setattr(self, key, value)
        self.question = question
        if 'llama2' in self.policy_type or 'abel' in self.policy_type:
            # trained policy: w/o few-shot examples, and with IMPROVE_LLAMA_POLICY_PROMPT
            if 'improved' in self.policy_type:
                question_wo_fewshot = question.split('\n\n')[-1]
                self.question_for_policy = IMPROVE_LLAMA_POLICY_PROMPT.format(query=question_wo_fewshot)
                self.question_for_simulate_policy = question
            else:
                self.question_for_policy = question
                self.question_for_simulate_policy = question
            # question_for_policy: "8-shots\n\nQuestion: xxx\nAnswer:"
            self.question_for_value = question.split('\n\n')[-1]  # remove few-shot
            # question_for_value: "Question: xxx\nAnswer:"
        elif 'wizardmath' in self.policy_type:
            self.question_for_policy = WIZARD_MATH_PROMPT.format(instruction=question) + "\n\n"
            self.question_for_value = question
            self.question_for_simulate_policy = self.question_for_policy
        elif 'llama3' in self.policy_type:
            self.question_for_policy = LLAMA3_PROMPT.format(QUERY=question)
            self.question_for_value = question
            self.question_for_simulate_policy = self.question_for_policy
        self.answer = answer  # ground truth
        self.answer_number = self.extract_answer(answer, self.dataset, self.policy_type, is_reference=True)
        self.cur_answer = cur_answer
        self.max_lines = max_lines
        self.cur_step = cur_step
        self.config = config
        self.is_terminate = is_terminate
        self.cur_answer_w_math_extractor = cur_answer_w_math_extractor
        self.has_code_exec = has_code_exec
        self.step_anser = None

    def get_cur_answer(self):
        return self.cur_answer

    def get_answer_number(self):
        return self.answer_number

    def get_pred_answer_number(self):
        return self.extract_answer(self.cur_answer, self.dataset, self.policy_type, is_reference=False)

    def get_cur_step(self):
        return self.cur_step

    def add_calculator_for_trajecotry(self, cur_answer):
        """
        Add calculator for each step in the trajectory
        """
        if self.dataset == 'gsm8k':
            split_sign = "\n"
            add_calculator = GSM8KDataset.add_calculator
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        cur_answer_step = cur_answer.split(split_sign)
        n_current_steps = len(cur_answer_step)

        answer_w_calculator = ""
        for i in range(n_current_steps):
            if len(cur_answer_step[i]) == 0:
                continue
            step_w_calculator = add_calculator(cur_answer_step[i])
            answer_w_calculator += step_w_calculator
            if i < n_current_steps - 1 or not self.check_answer_exist(cur_answer_step[i]):
                # do not add split sign for the last step
                answer_w_calculator += split_sign
        return answer_w_calculator

    def get_value(self, is_additional, path=None):
        """
        MLP classification task.
        """
        if path is not None:
            cur_answer = path
        else:
            cur_answer = self.cur_answer

        if self.use_calculator:
            cur_answer = self.add_calculator_for_trajecotry(cur_answer)

        if is_additional:
            value_url = select_url(self.add_value_url)
        else:
            value_url = select_url(self.value_url)

        assert len(cur_answer) > 0, f"Cur answer must be longer than 0! This should not be root"

        cur_answer = cur_answer.replace('<|eot_id|>', '')
        prompt = VALUE_INSTRUCTION + self.question_for_value + cur_answer

        return get_value(value_url=value_url, prompt=prompt)

    def get_value_70b(self, is_additional):
        """
        Request the vllm server for 70b model
        NOTE: we train value net without using the '\n' at the end, let's remove it first.
              and do not remove the beginning space ' '.
        """
        if is_additional:
            value_url = self.add_value_url
        else:
            value_url = self.value_url
        return get_value_70b(value_url=value_url,
                             question_for_value=self.question_for_value,
                             cur_answer=self.cur_answer)

    def get_step_reward(self):
        """
        NOTE: We don't have end of trajctory information in training, thus,
              we train step reward without using the '\n' at the end, let's remove it first.
              and do not remove the beginning space ' '.
        """
        return get_step_reward(step_reward_url=self.step_reward_url,
                               question_for_value=self.question_for_value,
                               cur_answer=self.cur_answer)

    def get_step_reward_70b(self):
        cur_answer = self.cur_answer_w_math_extractor if self.use_math_extractor else self.cur_answer
        cur_answer = cur_answer.rstrip()
        assert len(cur_answer) > 0, f"Cur answer must be longer than 0! This should not be root"

        instruction = STEP_CAL_INSTRUCTION if self.use_math_extractor else STEP_INSTRUCTION

        prompt = f"{instruction}{self.question_for_value}{cur_answer}\n\n{STEP_RESPONSE}"
        true_text = "True"
        false_text = "False"
        return get_step_reward_70b(self.step_reward_url, prompt, true_text, false_text)

    def get_outcome_70b(self, answer2end):
        answer2end = answer2end.rstrip()
        assert len(answer2end) > 0, f"Cur answer must be longer than 0! This should not be root"

        prompt = f"{OUTCOME_INSTRUCTION}{self.question_for_value}{answer2end}\n\n{OUTCOME_RESPONSE}"
        tokenizer_type = 'llama2'
        return get_outcome_70b(orm_url=self.orm_url,
                               prompt=prompt,
                               orm_calibrate_logits=self.orm_calibrate_logits,
                               tokenizer_type=tokenizer_type)

    def get_math_extractor(self, step):
        return get_math_extractor(step_cal_url=self.math_extractor_url, step=step)

    def get_value_prompt_batch(self, prompts):
        payload = {"texts": prompts}  # Use the list of prompts directly
        if DEBUG:
            print('get_value_prompt_batch', payload)
        response = requests.post(self.value_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            if DEBUG:
                print("Values:", result["values"])
            return result["values"]
        else:
            if DEBUG:
                print("Values Error:", response.status_code, response.text)
            return None

    def check_step_extend_rule_based(self, output):
        """
        Rule-based method to determine whether to continue generating steps instead of just one: 
        For the WizardMath model, the first step sometimes contains only phrases like "First, we are given the function:" 
        In such cases, it is better to generate more steps to provide more information.
        """
        # if self.cur_step == 0 and output[-1] == ":":
        #     return not bool(find_equations_in_text(output))
        # return False
        return not bool(find_equations_in_text(output))

    def cond_actions(self, is_simulation=False, to_end=False, is_greedy=False, avoid_empty=False):
        """
        Conditional actions. If avoid_empty and the action is empty, then re-sample.
        """
        n_attempts = 20
        for attempt in range(n_attempts):
            try:
                if to_end:
                    action, has_end_token = self.action2end(is_simulation=is_simulation, is_greedy=is_greedy)
                else:
                    action, has_end_token = self.actions(is_simulation=is_simulation, is_greedy=is_greedy)

                if avoid_empty and len(action.strip()) == 0:
                    return self.cond_actions(is_simulation, to_end, False, avoid_empty)
                return action, has_end_token
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f'ip: {self.policy_url}, attempt {attempt}, Type of error: {type(e).__name__}', flush=True)
                    continue
                else:
                    print(f'ip: {self.policy_url}, Ending. Type of error: {type(e).__name__}', flush=True)
                    raise e

    def actions(self, is_simulation=False, is_greedy=False):
        """
        Only predict for next step.
        """
        openai.api_key = "ss"
        openai.api_base = select_url(self.policy_url)
        policy_model_type = self.policy_type
        question = self.question_for_policy

        if 'llama3' in policy_model_type:
            stop = ['<|eot_id|>', '\n\n\n', '\n\n', '\n', 'Question', '\n\nQuestion:']  # should in this order
            step_sign = ''
            temperature = 1.0 if not is_greedy else 0.0
        elif "llama" in policy_model_type or 'abel' in policy_model_type:
            stop = ['\n\n', '\n', 'Question', '\n\nQuestion:']  # should in this order
            step_sign = '\n'
            temperature = 0.8 if not is_greedy else 0.0
        elif "wizardmath" in policy_model_type:
            stop = ['\n\n']
            step_sign = '\n\n'
            temperature = 1.0 if not is_greedy else 0.0
        else:
            raise ValueError(f"Unknown policy model type: {policy_model_type}")

        prefix_qa = f"{question}{self.cur_answer}"  # root: cur_answer is empty, other: cur_answer starts with " "
        action, has_end_token = self.request_actions(policy_model_type, prefix_qa, temperature, stop)

        if DEBUG:
            print('actions', action)

        if self.use_rule_based_step_extend and self.check_step_extend_rule_based(action) and not has_end_token:
            prefix_qa = f"{question}{self.cur_answer}{action}{step_sign}"  # the step_sign is included in the action
            action_extend, has_end_token = self.request_actions(policy_model_type, prefix_qa, temperature, stop)
            action = f"{action}{step_sign}{action_extend}"

        # here we can directly check if action contains code pattern, as the code and solution pattern are always in the same step
        if self.exec_code and not self.has_code_exec and check_code_pattern(action):
            action = code_api_remove_leftover(action)
            self.has_code_exec = True
            prefix_qa = f"{question}{self.cur_answer}{action}"
            action_append, has_end_token = self.request_actions(policy_model_type, prefix_qa, temperature, stop)
            action += action_append

        return action, has_end_token

    def request_actions(self, policy_model_type, prefix_qa, temperature, stop):
        has_end_token = False
        if "abel" in policy_model_type:
            prefix_qa = [{'role': "user", 'content': prefix_qa}]
            completion = openai.ChatCompletion.create(model=self.policy_model,
                                                      messages=prefix_qa,
                                                      temperature=temperature,
                                                      top_p=1.0,
                                                      max_tokens=1024,
                                                      logprobs=1,
                                                      n=1,
                                                      include_stop_str_in_output=True,
                                                      stop=stop)
            action = completion.choices[0]['message']['content']
        else:
            completion = openai.Completion.create(model=self.policy_model,
                                                  prompt=prefix_qa,
                                                  temperature=temperature,
                                                  top_p=1.0,
                                                  max_tokens=1024,
                                                  logprobs=1,
                                                  n=1,
                                                  include_stop_str_in_output=True,
                                                  stop=stop)
            action = completion.choices[0]['text']
            last_token = completion.choices[0]['logprobs']['tokens'][-1]
            if "wizardmath" in policy_model_type and last_token == STOP_TOKEN_WIZARDMATH:
                has_end_token = True  # this is only useful for wizardmath
            if 'llama3' in policy_model_type and last_token == STOP_TOKEN_LLAMA3:
                has_end_token = True  # this is only useful for llama3

        return action, has_end_token

    def action2end(self, is_simulation=False, is_greedy=False):
        """
        Predict to end for complete trajectory.
        """
        assert is_simulation == False  # "Simulation is not supported for action2end". Should use action2end_batch for simulation
        openai.api_key = "ss"
        openai.api_base = select_url(self.policy_url)
        policy_model_type = self.policy_type
        question = self.question_for_policy

        if 'llama3' in policy_model_type:
            stop = ['<|eot_id|>']
        elif "llama" in policy_model_type or "abel" in policy_model_type:
            if self.dataset == 'gsm8k':
                stop = ['\n\nQuestion:', '\n\n', 'Question:', '\n\nAnswer:']
            elif self.dataset == 'math':
                stop = ['\nUser:', '\nAssistant:']
            elif self.dataset == 'jiping':
                stop = ['<|eot_id|>']
        elif "wizardmath" in policy_model_type:
            stop = ["Instruction:", "Instruction", "Response:", "Response"]
        else:
            raise ValueError(f"Unknown policy model type: {policy_model_type}")

        temperature = 0.8 if not is_greedy else 0.0
        prefix_qa = f"{question}{self.cur_answer}"  # root: cur_answer is empty, other: cur_answer starts with " "
        action, has_end_token = self.request_actions(policy_model_type, prefix_qa, temperature, stop)

        # here we can directly check if action contains code pattern, as the code and solution pattern are always in the same step
        if self.exec_code and not self.has_code_exec and check_code_pattern(action):
            action = code_api_remove_leftover(action)
            self.has_code_exec = True
            prefix_qa = f"{question}{self.cur_answer}{action}"
            action_append, has_end_token = self.request_actions(policy_model_type, prefix_qa, temperature, stop)
            action += action_append

        if DEBUG:
            print('action2end', action)
        return action, has_end_token

    def action2end_batch(self, is_simulation=False, batch_size=1, cur_answer=None):
        openai.api_key = "ss"

        n_attempts = 10
        for attempt in range(n_attempts):
            try:
                if is_simulation:
                    openai.api_base = select_url(self.simulate_policy_url)
                    policy_model_type = self.simulate_policy_type
                    policy_model = self.simulate_policy_model
                    question = self.question_for_simulate_policy
                else:
                    openai.api_base = select_url(self.policy_url)
                    policy_model_type = self.policy_type
                    policy_model = self.policy_model
                    question = self.question_for_policy

                if 'llama3' in policy_model_type:
                    stop = ['<|eot_id|>']
                elif "llama" in policy_model_type or "abel" in policy_model_type:
                    stop = ['\n\nQuestion:', '\n\n', 'Question:', '\n\nAnswer:']
                elif "wizardmath" in policy_model_type:
                    stop = ['\nUser:', '\nAssistant:']
                else:
                    raise ValueError(f"Unknown policy model type: {policy_model_type}")

                if cur_answer is not None:
                    prefix_qa = question + cur_answer
                else:
                    prefix_qa = question + self.cur_answer  # root: cur_answer is empty, other: cur_answer starts with " "
                if "abel" in policy_model_type:
                    prefix_qa = [{'role': "user", 'content': prefix_qa}]
                    completions = openai.ChatCompletion.create(model=policy_model,
                                                               messages=prefix_qa,
                                                               t=1.0,
                                                               n=batch_size,
                                                               top_p=1.0,
                                                               max_tokens=1024,
                                                               stop=stop)
                    results = [choice['message']['content'].lstrip() for choice in completions.choices]
                    # the abel-002 will add a \n in the beinging. we need to remove it
                else:
                    completions = CustomCompletion.create(
                        model=policy_model,
                        prompt=prefix_qa,
                        n=batch_size,
                        temperature=1.0,
                        top_p=1.0,
                        max_tokens=1024,  # 512
                        stop=stop)
                    results = [choice.text for choice in completions.choices]
                if DEBUG:
                    for i, result in enumerate(results):
                        print(f'action2end_batch {i}:', result)
                return results
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f'ip: {self.policy_url}, attempt {attempt}, Type of error: {type(e).__name__}', flush=True)
                    continue
                else:
                    print(f'ip: {self.policy_url}, Ending. Type of error: {type(e).__name__}', flush=True)
                    raise e

    def take_action(self, action, has_end_token):
        answer_pattern_exists = self.check_answer_exist(action)
        policy_model_type = self.policy_type
        answer_updated = self.cur_answer + action
        # add the math extractor calc at for the new generated action
        if self.use_math_extractor:
            answer_w_math_extractor_updated = self.cur_answer_w_math_extractor + self.get_math_extractor(action)
        else:
            answer_w_math_extractor_updated = ""
        # the step sign is included in the action, no need to add it here
        if 'llama3' in policy_model_type:
            pass
        elif "llama" in policy_model_type or "abel" in policy_model_type:
            if not answer_pattern_exists and not has_end_token:
                # this is a prefix answer
                answer_updated += "\n"
                answer_w_math_extractor_updated += "\n"
        elif "wizardmath" in policy_model_type:
            if not has_end_token:
                answer_updated += '\n\n'
                answer_w_math_extractor_updated += '\n\n'
        else:
            raise ValueError(f"Unknown policy model type: {policy_model_type}")

        if DEBUG:
            print('take_action', answer_updated)

        is_terminate = answer_pattern_exists or has_end_token
        next_state = ProblemState(self.question,
                                  self.answer,
                                  self.config,
                                  cur_answer=answer_updated,
                                  cur_step=self.cur_step + 1,
                                  max_lines=self.max_lines,
                                  is_terminate=is_terminate,
                                  cur_answer_w_math_extractor=answer_w_math_extractor_updated,
                                  has_code_exec=self.has_code_exec)
        return next_state

    def take_action_end(self, is_simulation):
        if self.is_terminal_real():
            return self
        n_attempts = 20
        for attempt in range(n_attempts):
            try:
                action, _ = self.action2end(is_simulation=is_simulation)
                break
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f'ip: {self.policy_url}, attempt {attempt}, Type of error: {type(e).__name__}', flush=True)
                    continue
                else:
                    print(f'ip: {self.policy_url}, Ending. Type of error: {type(e).__name__}', flush=True)
                    raise e
        # action, _ = self.action2end(is_simulation=is_simulation)
        policy_model_type = self.policy_type
        if 'llama3' in policy_model_type:
            action_split = action.split('\n')
            action_split = [a for a in action_split if len(a) > 0]  # llama3 has a mix of \n and \n\n
            n_steps = len(action_split)
        elif "llama" in policy_model_type or "abel" in policy_model_type:
            n_steps = len(action.split('\n'))
        elif "wizardmath" in policy_model_type:
            n_steps = len(action.split('\n\n'))
        else:
            raise ValueError(f"Unknown policy model type: {policy_model_type}")
        answer_updated = self.cur_answer + action
        if self.use_math_extractor:
            answer_w_math_extractor_updated = self.cur_answer_w_math_extractor + self.get_math_extractor(action)
        else:
            answer_w_math_extractor_updated = ""
        if DEBUG:
            print('take_action_end', answer_updated)
        end_state = ProblemState(self.question,
                                 self.answer,
                                 self.config,
                                 cur_answer=answer_updated,
                                 cur_step=self.cur_step + n_steps,
                                 max_lines=1000,
                                 is_terminate=True,
                                 cur_answer_w_math_extractor=answer_w_math_extractor_updated,
                                 has_code_exec=self.has_code_exec)
        return end_state

    def take_action_rewarad_batch(self, is_simulation, batch_size, use_orm=True):
        """
        Fast-rollout to end, and return all values
        """
        if self.dataset == 'math':
            action_list = []
            for i in range(batch_size):
                action_list += self.action2end_batch(is_simulation=is_simulation, batch_size=1)
        else:
            action_list = self.action2end_batch(is_simulation=is_simulation, batch_size=batch_size)

        if DEBUG:
            print('fastrollout action_list', action_list)
        prefix_prev_a = f"{self.cur_answer}"
        if self.exec_code and not self.has_code_exec:
            reward_list = []
            for action in action_list:
                answer_updated = f"{prefix_prev_a}{action}"
                if check_code_pattern(answer_updated):
                    answer_updated = code_api_remove_leftover(answer_updated)
                    action = self.action2end_batch(is_simulation=is_simulation, batch_size=1, cur_answer=answer_updated)
                    answer_updated += action[0]
                    # print ('In fast-rollout', [answer_updated])
                reward = self.get_outcome_70b(answer_updated)
                reward_list.append(reward)
        else:
            if use_orm:
                prompt_updated_action_list = [f"{prefix_prev_a}{action}" for action in action_list]
                reward_list = [self.get_outcome_70b(action) for action in prompt_updated_action_list]
            else:
                prefix_qa = f"{self.question_for_value}{self.cur_answer}"  # ends with \n for partial ans
                prompt_updated_action_list = [f"{prefix_qa}{action}" for action in action_list]
                reward_list = self.get_value_prompt_batch(prompt_updated_action_list)
        return reward_list

    def is_terminal(self):
        answer_pattern_exists = self.check_answer_exist(self.cur_answer)
        max_lines_reached = self.cur_step >= self.max_lines
        if DEBUG:
            print('is_terminal', answer_pattern_exists, max_lines_reached)
        return answer_pattern_exists or max_lines_reached or self.is_terminate

    def is_terminal_real(self):
        answer_pattern_exists = self.check_answer_exist(self.cur_answer)
        if DEBUG:
            print('is_terminal', answer_pattern_exists)
        return answer_pattern_exists or self.is_terminate

    def check_answer_exist(self, answer):
        if self.dataset == 'gsm8k' and 'llama' in self.policy_type:
            match = ANS_RE.search(answer)
            if match:
                return True
            else:
                return False
        elif self.dataset == 'math':
            return False
        elif self.dataset == 'jiping':
            return False
        else:
            raise ValueError(f"Dataset {self.dataset} and policy type {self.policy_type} pair currently not supported")

    @staticmethod
    def extract_answer(completion, dataset, policy_type, is_reference=False):
        completion = completion.replace(STOP_TOKEN_LLAMA3, "")
        completion = completion.replace(STOP_TOKEN_WIZARDMATH, "")
        if dataset == 'gsm8k' and 'llama' in policy_type:
            match = ANS_RE.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return INVALID_ANS
        else:
            raise ValueError(f"Dataset {dataset} and policy type {policy_type} pair currently not supported")

    def is_correct(self, output_score=False):
        gt_answer = self.answer_number
        assert gt_answer != INVALID_ANS
        extracted_anwer = self.extract_answer(self.cur_answer, self.dataset, self.policy_type, is_reference=False)
        # Align with the evaluation in value training
        if extracted_anwer == gt_answer:
            return True
        else:
            try:
                if eval(extracted_anwer) == eval(gt_answer):
                    return True
                else:
                    return False
            except:
                return False


class MonteCarloTreeSearchNode():

    def __init__(self,
                 state,
                 config,
                 parent=None,
                 parent_action=None,
                 depth=0,
                 traj_step_reward=None,
                 node_id=None,
                 n_repeat_by_parent=1):

        for key, value in config.items():
            setattr(self, key, value)
        self.config = config
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = []  # list of back propagated values
        self._traj_step_reward = traj_step_reward if traj_step_reward is not None else []  # list of step reward

        self._values = []  # raw values
        self._outcome_rewards = []  # raw values from orm
        self._outcome_rewards_list = []
        self._additional_values = []  # raw values from additional value net
        self._cached_reward = 0.
        self.depth = depth
        self.n_repeat = 0
        if self.max_split_depth < 0:  # disable the maximum depth to split the tree mechanism
            self.max_split_depth = self.depth
        if self.depth == 0:
            # set the piror number of children for the root node
            self.n_total_children_adaptive = self.init_children if self.init_children > -1 else self.n_total_children
        elif self.depth > self.max_split_depth:
            self.n_total_children_adaptive = 1
        else:
            self.n_total_children_adaptive = self.n_total_children
        self.max_q_diff = 0
        self.node_id = node_id
        self.n_repeat_by_parent = n_repeat_by_parent  # n of repeats when generating this node

    def retrieve_origin_value(self):
        return self._values[0] if len(self._values) > 0 else None

    def set_cached_reward(self, reward):
        self._cached_reward = reward

    def get_cached_reward(self):
        return self._cached_reward

    def get_details(self):
        """
        Return a dict for detailed information about itself
        """
        return {
            "q": round(self.q(), 2),
            "n": self.n(),
            "bp_values": [round(x, 2) for x in self._results],
            "value_reward": [round(x, 2) for x in self._values],
            # "traj_step_reward": [round(x, 2) for x in self._traj_step_reward],
            "outcome_reward": [round(x, 2) for x in self._outcome_rewards],
            # "outcome_rewards_list": [round(x, 2) for x in self._outcome_rewards_list] +
            #                         [np.var(np.array(self._outcome_rewards_list))],
            "outcome_rewards_list": [round(x, 2) for x in self._outcome_rewards_list],
            # "additional_value": [round(x, 2) for x in self._additional_values],
            "n_children": self.n_children(),
            "n_repeat_by_parent": self.n_repeat_by_parent,
            "is_terminal_node": self.is_terminal_node(),
        }

    def total_number_nodes(self):
        """
        Get total number of nodes in a tree.
        """
        tot_node = 1  # itself
        for child in self.children:
            tot_node += child.total_number_nodes()
        return tot_node

    def n(self):
        return self._number_of_visits

    def q(self):
        return np.sum(self._results)

    def add_step_reward(self):
        if '70b' in self.step_reward_type:
            step_reward = self.get_step_reward_70b()
        else:
            step_reward = self.get_step_reward()
        self._traj_step_reward.append(step_reward)
        if DEBUG:
            print(f"traj_step_reward: {self._traj_step_reward}, depth: {self.depth}")
        assert len(self._traj_step_reward) == self.depth

    def add_value(self, is_additional=False):
        if is_additional:
            value_type = self.add_value_type
        else:
            value_type = self.value_type

        if '70b' in value_type:
            assert 'lm' in value_type  # currently we only support lm model for 70b
            raw_value = self.get_value_70b(is_additional)
            raw_value = (raw_value - 0.5) * 2  # normalize to [-1, 1] to be consistent with regression value
        else:
            raw_value = self.get_value(is_additional)

        if is_additional:
            self._additional_values.append(raw_value)
        else:
            self._values.append(raw_value)
        return raw_value

    def n_children(self):
        return len(self.children)

    def is_terminal_node(self):
        return self.state.is_terminal()

    def is_real_terminal_node(self):
        return self.state.is_terminal_real()

    def is_fully_expanded(self):
        return self.n_children() >= self.n_total_children_adaptive

    def get_acceptable_action(self):

        def edit_distance(s1, s2):
            m = len(s1)
            n = len(s2)
            dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
            return dp[m][n]

        def get_min_edit_distance(s, actions):
            min_dist = 10000
            for action in actions:
                dist = edit_distance(s, action)
                if dist < min_dist:
                    min_dist = dist
            return min_dist

        def get_model_based_similarity_rate(s, actions, question, previous_step):
            rst = []
            for action in actions:
                is_similar = self.get_model_based_action_similarities(s, action, question, previous_step)
                # print (f"get_model_based_similarity_rate: {s}, {action}, {is_similar}")
                rst.append(is_similar)
            return np.mean(rst)

        children_actin_list = [child.parent_action for child in self.children]
        n_repeat = 0

        # if depth reaches max depth, rollout to terminal.
        to_end = self.max_depth <= (self.depth + 1)

        # if greedy_path and no children, use greedy to generate.
        is_greedy = self.greedy_path and len(self.children) == 0
        # avoid_empty for the first layer
        avoid_empty = self.depth == 0

        if to_end:
            reason = f"Reaching the max depth {self.max_depth}" if self.max_depth <= (
                self.depth + 1) else "Model is pretty sure the node is correct"
            print(f"{reason}, current step {self.depth}, value {self.retrieve_origin_value()}, rollout to terminal.")
        action = ""
        if self.state_merge_method == 'edit_distance':
            min_dist = -1
            action_list = []
            min_dist_list = []
            while min_dist < self.edit_dist_thres and n_repeat < self.max_n_repeat:
                action, has_end_token = self.state.cond_actions(to_end=to_end,
                                                                is_greedy=is_greedy,
                                                                avoid_empty=avoid_empty)
                min_dist = get_min_edit_distance(action, children_actin_list)
                action_list.append(action)
                min_dist_list.append(min_dist)
                if DEBUG:
                    print('get_acceptable_action', n_repeat, action, min_dist)
                n_repeat += 1

                if ANS_RE.search(action) is not None:
                    break
            # here we choose the action with the max of minimum edit distance
            max_min_dist_index = np.argmax(min_dist_list)
            action = action_list[max_min_dist_index]

        elif self.state_merge_method == 'model_based':
            similar_rate = 1.0
            while similar_rate > self.similarity_thres and n_repeat < self.max_n_repeat:
                action, has_end_token = self.state.cond_actions(to_end=to_end,
                                                                is_greedy=is_greedy,
                                                                avoid_empty=avoid_empty)
                if len(children_actin_list) == 0:
                    break
                question = self.state.question_for_value
                previous_step = self.state.cur_answer
                similar_rate = get_model_based_similarity_rate(action, children_actin_list, question, previous_step)
                if DEBUG:
                    print('get_acceptable_action', n_repeat, action, similar_rate)
                n_repeat += 1

                if ANS_RE.search(action) is not None:
                    break
        elif self.state_merge_method == 'None' or self.state_merge_method == None:
            action, has_end_token = self.state.cond_actions(to_end=to_end, is_greedy=is_greedy, avoid_empty=avoid_empty)
        else:
            raise ValueError(f"Unknown state merge method: {self.state_merge_method}")
        return action, has_end_token, n_repeat

    def expand(self):
        action, has_end_token, n_repeat = self.get_acceptable_action()
        self.n_repeat = n_repeat
        next_state = self.state.take_action(action, has_end_token)
        cur_n_children = len(self.children)
        cur_node_id = self.node_id
        child_node = MonteCarloTreeSearchNode(next_state,
                                              config=self.config,
                                              parent=self,
                                              parent_action=action,
                                              depth=self.depth + 1,
                                              traj_step_reward=copy.copy(self._traj_step_reward),
                                              node_id=f"{cur_node_id}-{cur_n_children}",
                                              n_repeat_by_parent=n_repeat)
        self.children.append(child_node)
        child_index = len(self.children) - 1
        ancestor_child_indices = child_node.get_ancestor_child_indices()
        return child_node

    def update_n_total_children(self, increase_factor):
        if not self.children:
            return
        values = [c.q() / c.n() for c in self.children]
        # value_diff = max(values) - min(values)  # currently use max_q - min_q. Will change to max(abs(q - mean(q)) later
        values = np.array(values)
        mean_value = np.mean(values)
        diff_values = np.abs(values - mean_value)
        value_diff = np.max(diff_values)
        if value_diff > self.max_q_diff:
            self.max_q_diff = value_diff

        new_n_total_children = min(int(increase_factor * value_diff), 10)
        if new_n_total_children > self.n_total_children_adaptive:
            self.n_total_children_adaptive = new_n_total_children
        # elif self.depth == 0: # search more when mean value of first layer is negative. Currently deactivated.
        #     mean_value = np.mean(values)
        #     if mean_value < 0:
        #         self.n_total_children_adaptive = min(self.n_total_children_adaptive + 1, 10)

        # if self.stop_repeat_node_thres is no larger than self.max_n_repeat and self.n_repeat reaches the threshold
        # then stop expanding the node
        if self.n_repeat >= self.stop_repeat_node_thres:
            self.n_total_children_adaptive = max(1, len(self.children))
            if DEBUG:
                print(
                    f"Stop expanding node at depth {self.depth}, n_repeat {self.n_repeat}, n_total_children_adaptive {self.n_total_children_adaptive}"
                )
        # when stop_expansion_rollout_var > 0, if the variance of the fast rollout is smaller than the threshold
        # then stop expanding the node
        if self.stop_expansion_rollout_var > 0 and self.n_simulations > 0 and self.fastrollout_weight > 0:
            fastrollout_var = np.var(np.array(self._outcome_rewards_list))
            if fastrollout_var < self.stop_expansion_rollout_var:
                self.n_total_children_adaptive = max(1, len(self.children))

    def get_ancestor_child_indices(self):
        indices = []
        current_node = self
        while current_node.parent is not None:
            index = current_node.parent.children.index(current_node)
            indices.append(index)
            current_node = current_node.parent
        return indices[::-1]

    def get_value(self, is_additional):
        return self.state.get_value(is_additional)

    def get_value_70b(self, is_additional):
        return self.state.get_value_70b(is_additional)

    def get_step_reward(self):
        return self.state.get_step_reward()

    def get_step_reward_70b(self):
        return self.state.get_step_reward_70b()

    def get_outcome_70b_terminal_state(self):
        assert self.is_real_terminal_node()
        outcome_reward = self.state.get_outcome_70b(self.state.cur_answer)
        self._outcome_rewards.append(outcome_reward)
        return outcome_reward

    def get_model_based_action_similarities(self, action_0, action_1, question, previous_step):
        state_merge_model = self.state_merge_model
        openai.api_base = select_url(self.state_merge_url)
        openai.api_key = "ss"

        # It is not necessary to add the \n
        action_0 = previous_step + action_0
        action_1 = previous_step + action_1

        prompt = MODEL_MERGE_TEMPLATE.format(question=question,
                                             action_0=action_0,
                                             action_1=action_1,
                                             instruction=MODEL_MERGE_INSTRUCTION)
        prompt_llama = format_llama2(prompt)
        completion = openai.ChatCompletion.create(model=state_merge_model,
                                                  messages=prompt_llama,
                                                  temperature=0.00,
                                                  max_tokens=2)
        if 'yes' in completion.choices[0]['message']['content'] or 'Yes' in completion.choices[0]['message']['content']:
            return 1
        else:
            return 0

    def add_simulate(self, n_simulations):
        """
        Add simulation: call simulate, then add to values
        """
        mean_value, reward_list = self.simulate(n_simulations=n_simulations)
        self._outcome_rewards.append(mean_value)
        self._outcome_rewards_list = reward_list
        return mean_value

    def simulate(self, n_simulations):
        """
        Fast-rollout and return mean value from ORM.
        """
        current_rollout_state = self.state
        reward_list = current_rollout_state.take_action_rewarad_batch(is_simulation=True,
                                                                      batch_size=n_simulations,
                                                                      use_orm=True)
        if DEBUG:
            print(f"Simulation reward_list: {reward_list}")
        final_value = np.mean(reward_list)
        return final_value, reward_list

    def get_end_state(self):
        end_state = self.state.take_action_end(is_simulation=False)
        return end_state

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results.append(result)
        if self.parent:
            self.parent.backpropagate(result)

    def best_child(self, prune_low_value=False):
        """
        Picking the best child based on UCT: w + c * sqrt(log(self.n) / child.n)
        """
        # currently don't add 2 * np.log(self.n()) / c.n()
        choices_weights = [(c.q() / c.n()) + self.c_param * np.sqrt((np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def _tree_policy(self):
        """
        Select and expand
        Selection strategy: if not fully expanded, pick current node, otherwise pick best child and check
        """
        parent_node = self
        current_node = self
        is_expand = False
        while not current_node.is_terminal_node():
            current_node.update_n_total_children(self.width_increase_factor)
            if not current_node.is_fully_expanded():
                is_expand = True
                return current_node.expand(), is_expand
            else:
                parent_node = current_node
                current_node = current_node.best_child(prune_low_value=self.config["low_value_prune"])
            # lfsong: if all children are low value, return the parent for update
            if current_node == None:
                return parent_node, False
        return current_node, is_expand

    def best_child_greedy(self):
        if not self.children:
            return None
        choices_weights = [c.q() / c.n() for c in self.children]
        choice_index = np.argmax(choices_weights)
        if DEBUG:
            print(f"layer: {self.depth}, q_dist: {[f'{x:.2f}' for x in choices_weights]}, choice: {choice_index}")
        return self.children[choice_index]

    def best_action_greedy(self):
        current_node = self.best_action_greedy_leaf()
        final_state = current_node.get_end_state()
        return final_state

    def best_action_greedy_leaf(self):
        """
        Find the leaf with greedy q/n
        Return: MonteCarloTreeSearchNode
        """
        current_node = self
        while not current_node.is_terminal_node():
            next_node = current_node.best_child_greedy()
            if next_node is None:
                break
            current_node = next_node
        return current_node

    def generate_all_paths(self):
        all_paths = []
        all_path_set = set()
        queue = deque((x, 1) for x in self.children)
        while len(queue) > 0:
            cur, cur_depth = queue.popleft()
            cur_path = cur.state.cur_answer
            if cur_path in all_path_set:
                continue
            all_paths.append({
                "path": cur_path,
                "depth": cur_depth,
                "score": cur.get_cached_reward(),
                "is_terminal": cur.is_real_terminal_node(),
            })
            all_path_set.add(cur_path)
            queue.extend((x, cur_depth + 1) for x in cur.children)
        return all_paths

    # TODO (lfsong): add counts information here
    def generate_all_paths_with_rankings(self):
        all_paths = []
        for leaf in self.get_all_leaves():
            if not leaf.is_real_terminal_node():
                continue
            path = {
                "full_trajectory": leaf.state.cur_answer,
                "score": leaf.get_cached_reward(),
                "value_score": leaf.retrieve_origin_value(),
                "is_correct": None,  # leaf.state.is_correct(),
                "rankings": {}  # prefix --> rankings
            }
            cur_parent = leaf.parent.parent  # leaf.parent --> cur_node
            while cur_parent != None:
                cur_children = cur_parent.get_all_children_with_reward()
                if len(cur_children) > 1:
                    path["rankings"][cur_parent.state.cur_answer] = cur_children
                cur_parent = cur_parent.parent
            all_paths.append(path)
        return all_paths

    def generate_all_paths_with_preferences(self):
        all_paths = []
        tt_threshold = self.config.get("traversal_threshold", 0.5)
        cc_threshold = self.config.get("cc_threshold", 1.0)
        for leaf in self.get_all_leaves():
            if not leaf.is_real_terminal_node():
                continue
            path = {
                "full_trajectory": leaf.state.cur_answer,
                "score": leaf.get_cached_reward(),
                "value_score": leaf.retrieve_origin_value(),
                "is_correct": leaf.state.is_correct(),
                "preferences": []
            }
            cur_node = leaf.parent
            while cur_node != None and cur_node.parent != None:
                cur_sib = cur_node.get_sibling_max()
                if cur_sib != None:
                    pair = {
                        "prefix": cur_node.parent.state.cur_answer,
                        "a": cur_node.state.cur_answer,
                        "a_score": cur_node.get_cached_reward(),
                        "b": cur_sib.state.cur_answer,
                        "b_score": cur_sib.get_cached_reward()
                    }
                    path["preferences"].append(pair)

                cur_node = cur_node.parent

            cur_node, par_node = leaf, leaf.parent
            traversal_trajectory = cur_node.state.cur_answer[len(par_node.state.cur_answer):]

            cur_node = leaf.parent
            tt_count = 0
            while cur_node != None and cur_node.parent != None:
                par_node = cur_node.parent
                par_is_root = len(par_node.state.cur_answer) == 0
                cur_step_answer = cur_node.state.cur_answer[len(par_node.state.cur_answer):]
                traversal_trajectory = f"{cur_step_answer}{traversal_trajectory}"

                cur_score = cur_node.get_cached_reward()
                cur_sib = cur_node.get_sibling_min()

                if cur_sib != None and cur_score <= cc_threshold:
                    sib_score = cur_sib.get_cached_reward()
                    sib_children = cur_sib.get_all_children_with_reward()

                    if cur_score - sib_score >= tt_threshold and not par_is_root:
                        tt_count += 1
                        if len(sib_children) > 0:
                            # pick the lowest nephew
                            nep_lowest = sib_children[-1]
                            alt_step_answer = nep_lowest["answer"][len(par_node.state.cur_answer):]
                        else:
                            alt_step_answer = cur_sib.state.cur_answer[len(par_node.state.cur_answer):]

                        traversal_trajectory = f"\n\n### alternative thought start ###\n{alt_step_answer}\n### alternative thought end ###\n\n{traversal_trajectory}"

                cur_node = cur_node.parent
            path["traversal_trajectory"] = traversal_trajectory
            path["traversal_switch_count"] = tt_count

            all_paths.append(path)
        return all_paths

    def generate_index_path(self, curr):
        self.index_path = copy.copy(curr)
        for i, child in enumerate(self.children):
            child.generate_index_path(curr + [i])

    def get_sibling_max(self):
        """
        Return the max abs(diff) sibling node (may be the max or min)
        """
        if self.parent is None:
            return None
        diffs = [abs(sib.get_cached_reward() - self.get_cached_reward()) for sib in self.parent.children]
        if len(diffs) < 2 or max(diffs) <= 1e-3:
            return None
        idx = np.argmax(diffs)
        return self.parent.children[idx]

    def get_sibling_min(self):
        """
        Return the minimum sibling node (exclude self)
        """
        if self.parent is None:
            return None
        scores = [sib.get_cached_reward() for sib in self.parent.children]
        idx = np.argmin(scores)
        if abs(self.get_cached_reward() - scores[idx]) <= 1e-3:
            return None
        return self.parent.children[idx]

    def get_all_children_with_reward(self):
        if not self:
            return []
        return sorted([{
            "answer": x.state.cur_answer,
            "score": x.get_cached_reward()
        } for x in self.children],
                      key=lambda x: x["score"],
                      reverse=True)

    def get_all_leaves(self):
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves += child.get_all_leaves()
        return leaves

    # find the leaf with global best q/n
    def best_action_global_leaf(self):
        """
        Find the leaf with global best q/n
        Return: MonteCarloTreeSearchNode
        """
        if not self.children:
            return self
        best_leaf = None
        highest_average_q = float('-inf')
        for child in self.children:
            leaf = child.best_action_global_leaf()
            leaf_average_q = leaf.q() / leaf.n()
            if leaf_average_q > highest_average_q:
                highest_average_q = leaf_average_q
                best_leaf = leaf
        return best_leaf


class MCTS:

    def __init__(self, initial_state, config, args=None):
        self.initial_state = initial_state
        self.max_search_depth = 0
        self.args = args
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)
        self.unique_nodes = set()

    # set the root when loading the tree from pickle
    def set_root(self, root):
        self.root = root
        self.initial_state = root.state
        self.time_taken = 0
        self.update_internal_state()
        self.max_search_depth = max(len(eval(key)) for key in self.get_actions_dict().keys())
        self.unique_nodes = self.get_unique_nodes(self.root)
        self.total_rollouts = len(self.unique_nodes)
        self.total_steps, self.total_requests = self.cal_n_steps()

    def cal_n_steps(self):
        actions_dict = self.get_actions_dict()
        steps = 0
        requests = 0
        for key, value in actions_dict.items():
            if 'llama' in self.policy_type or 'abel' in self.policy_type:
                steps += len(value[0].split('\n'))
                requests += len(
                    value[0].split('\n')) * value[3]  # value[3] is the number of repeats to generate the action
            elif 'wizardmath' in self.policy_type:
                steps += len(value[0].split('\n\n'))
                requests += len(value[0].split('\n\n')) * value[3]
            else:
                raise ValueError(f"Unknown policy model type: {self.policy_type}")
        return steps, requests

    def get_unique_nodes(self, node=None, unique_nodes=None):
        if node is None:
            node = self.root
        if unique_nodes is None:
            unique_nodes = set()
        if node.children:
            for child in node.children:
                node_identifier = (child.depth, tuple(child.get_ancestor_child_indices()))  # Convert list to tuple
                unique_nodes.add(node_identifier)  # Add the unique identifier to the set
                self.get_unique_nodes(child, unique_nodes)
        return unique_nodes

    def run_mcts(self):
        """
        Core MCTS: Loop of selection, expanding, evaluation, and back-propagation
        """
        self.root = MonteCarloTreeSearchNode(state=self.initial_state, config=self.config, depth=0, node_id='root')
        n_steps, n_rollouts, n_requests, n_terminals = 0, 0, 0, 0
        search_time = 0
        while search_time < self.search_time or n_terminals < self.min_terminals:
            if search_time > (self.search_time * 10):
                break
            search_time += 1
            start_time = time.time()
            if self.args is not None and self.args.debug_log_level >= 4:
                print(f"Search {search_time} round\n\tStep of selection and expanding...")
            v, is_expand = self.root._tree_policy()
            if self.args is not None and self.args.debug_log_level >= 4:
                print(f"\tStep of evaluation...")
            if is_expand:
                reward = 0.0
                # the final reward is a weighted sum of value, fast rollout, and step reward
                # only calculate the cooresponding reward when the weight is larger than 0
                assert self.value_weight > 0 or (self.fastrollout_weight > 0 and
                                                 self.n_simulations > 0) or self.step_reward_weight > 0

                if self.value_weight > 0:
                    if self.args is not None and self.args.debug_log_level >= 4:
                        print(f"\t\tValue reward...")
                    reward += self.value_weight * v.add_value(is_additional=False)

                if self.add_value_weight > 0:
                    if self.args is not None and self.args.debug_log_level >= 4:
                        print(f"\t\tAdd value reward...")
                    reward += self.add_value_weight * v.add_value(is_additional=True)

                if self.n_simulations > 0 and self.fastrollout_weight:
                    if v.is_real_terminal_node():
                        if self.args is not None and self.args.debug_log_level >= 4:
                            print(f"\t\tOutcome reward...")
                        reward += self.fastrollout_weight * v.get_outcome_70b_terminal_state()
                    else:
                        if self.args is not None and self.args.debug_log_level >= 4:
                            print(f"\t\tSimulation reward...")
                        reward += self.fastrollout_weight * v.add_simulate(self.n_simulations)
                if self.step_reward_weight > 0:
                    if self.args is not None and self.args.debug_log_level >= 4:
                        print(f"\t\Step reward...")
                    v.add_step_reward()
                    if self.step_reward_mode == 'latest':  # use the last step reward
                        reward += self.step_reward_weight * v._traj_step_reward[-1]
                    elif self.step_reward_mode == 'avg':  # use the average step reward of the trajectory
                        reward += self.step_reward_weight * np.mean(v._traj_step_reward)
                    elif self.step_reward_mode == 'max':  # use the max step reward of the trajectory
                        reward += self.step_reward_weight * np.max(v._traj_step_reward)
                    elif self.step_reward_mode == 'min':  # use the min step reward of the trajectory
                        reward += self.step_reward_weight * np.min(v._traj_step_reward)
                    else:
                        raise ValueError(f"Unknown step reward mode: {self.step_reward_mode}")
                if self.args is not None and self.args.debug_log_level >= 4:
                    print(f"Starting backpropagation...")
                # cache reward
                v.set_cached_reward(reward)
                v.backpropagate(reward)
                end_time = time.time()
                if self.args is not None and self.args.debug_log_level >= 3:
                    print(f"Rollout: {n_rollouts}, "
                          f"Step: {n_steps}, "
                          f"action: {json.dumps(v.parent_action[-20:], ensure_ascii=False)}, depth: {v.depth}, "
                          f"ancestor: {v.get_ancestor_child_indices()}, "
                          f"n_child_allowed: {v.parent.n_total_children_adaptive}, "
                          f"max_q_diff: {v.parent.max_q_diff:.2f}, r: {reward:.2f}, "
                          f"node details: {v.get_details()}, "
                          f"time: {end_time - start_time:.2f}")
                parent_action = v.parent_action
                if 'llama' in self.policy_type or 'abel' in self.policy_type:
                    n_action_steps = len(parent_action.split('\n'))
                elif 'wizardmath' in self.policy_type:
                    n_action_steps = len(parent_action.split('\n\n'))
                # n_rollouts: num of nodes
                # n_steps: num of steps, as some node (in the end) may have multiple steps
                # n_requests: num of steps, consider the repeats in state merge
                n_steps += n_action_steps
                n_rollouts += 1
                n_requests += v.n_repeat_by_parent * n_action_steps
                if v.is_real_terminal_node():
                    n_terminals += 1
                node_identifier = (v.depth, tuple(v.get_ancestor_child_indices()))
                self.unique_nodes.add(node_identifier)
            else:
                if self.args is not None and self.args.debug_log_level >= 3:
                    print(f"\tSearched before - Depth: {v.depth}, ancestor: {v.get_ancestor_child_indices()}")
                reward = v.get_cached_reward()
                v.backpropagate(reward)

            if v.depth > self.max_search_depth:
                self.max_search_depth = v.depth
            if self.args is not None and self.args.debug_log_level >= 4:
                print(f"Search {search_time} round done")
        final_state = self.root.best_action_greedy()
        self.total_rollouts = n_rollouts
        self.total_steps = n_steps
        self.total_requests = n_requests
        return final_state

    def get_total_rollouts(self):
        return self.total_rollouts

    def get_total_requests(self):
        return self.total_requests

    def get_final_state_from_self_consistency(self):
        leaves = self.root.get_all_leaves()
        consis_groups = defaultdict(list)
        for i, leaf in enumerate(leaves):
            pred_number = leaf.state.get_pred_answer_number()
            if leaf.state.is_real_terminal_node() and pred_number == INVALID_ANS:
                print(f"INVALID_ANS: {leaf.state.get_cur_answer()}")
            else:
                consis_groups[pred_number].append(i)

        if len(consis_groups) == 0:
            return

        idx = sorted(consis_groups.values(), key=lambda x: -len(x))[0][0]
        selected_leaf = leaves[idx]
        selected_leaf_state = selected_leaf.get_end_state()
        return selected_leaf_state

    def get_final_state_from_global_topk(self, topk, keep_all_correct):
        self.root.generate_index_path(curr=[])
        leaves = sorted(self.root.get_all_leaves(), key=lambda x: -x.q() / x.n())

        all_correct = []
        global_top1_state = None
        for i in range(min(topk, len(leaves))):
            leaf_state = leaves[i].get_end_state()
            if leaf_state.is_correct():
                all_correct.append(leaf_state)
            if i == 0:
                global_top1_state = leaf_state

        if keep_all_correct:
            return all_correct
        else:
            return global_top1_state if all_correct == [] else all_correct[0]

    def get_final_state_from_global(self):
        """
        Find the leaf node w/ highest score.
        Then rollout to end to get the complete trajectory.
        """
        best_leaf = self.root.best_action_global_leaf()
        best_state = best_leaf.get_end_state()
        return best_state

    def get_all_paths(self):
        return self.root.generate_all_paths()

    def get_paths_with_alternatives(self, with_ranking=False):
        if with_ranking:
            return self.root.generate_all_paths_with_rankings()
        else:
            return self.root.generate_all_paths_with_preferences()

    def get_final_state_greedy(self):
        """
        Greedy search for the leaf node started from root node.
        Then rollout to end to get the complete trajectory.
        """
        greedy_leaf = self.root.best_action_greedy_leaf()
        greedy_state = greedy_leaf.get_end_state()
        return greedy_state

    def get_actions_dict(self, node=None, actions_dict=None):
        if node is None:
            node = self.root
        if actions_dict is None:
            actions_dict = {}
        if node.children:
            for child in node.children:
                ancestor_child_indices = child.get_ancestor_child_indices()
                q_value = child.q() / child.n()
                detailed_scores = child.get_details()
                actions_dict[str(ancestor_child_indices)] = (child.parent_action, q_value, detailed_scores,
                                                             child.n_repeat_by_parent)
                self.get_actions_dict(child, actions_dict)
        return actions_dict

    def get_best_action_path(self, node=None, path=None):
        if node is None:
            node = self.root
        if path is None:
            path = []
        if node.is_real_terminal_node():
            return path
        best_child = node.best_child_greedy()
        if best_child is not None:
            path.append(best_child.get_ancestor_child_indices())
            return self.get_best_action_path(best_child, path)
        return path

    def get_best_action_path_global(self):
        best_leaf = self.root.best_action_global_leaf()
        path = []
        current_node = best_leaf
        while current_node is not self.root:
            path.insert(0, current_node.get_ancestor_child_indices())
            current_node = current_node.parent
        return path

    def collect_terminal_answers(self, node):
        if node.is_real_terminal_node():
            return [node.state.get_pred_answer_number()], [node.q() / node.n()]
        terminal_answers = []
        terminal_values = []
        for child in node.children:
            answers, values = self.collect_terminal_answers(child)
            terminal_answers.extend(answers)
            terminal_values.extend(values)
        return terminal_answers, terminal_values

    def collect_terminal_nodes(self, node):
        # if node.is_real_terminal_node():
        if node.is_terminal_node():
            return [node]
        terminal_nodes = []
        for child in node.children:
            # print (child.state.cur_step, child.state.is_terminal(), child.state.max_lines)
            node_list = self.collect_terminal_nodes(child)
            terminal_nodes.extend(node_list)
        return terminal_nodes

    def collect_terminal_paths(self, node):
        # if node.is_real_terminal_node():
        if node.is_terminal_node():
            return [node.state.question_for_value + node.state.cur_answer]
        paths = []
        for child in node.children:
            path = self.collect_terminal_paths(child)
            paths.extend(path)
        return paths

    def update_internal_state(self, is_recompute_best_path=False):
        self.terminal_answers, self.terminal_values = self.collect_terminal_answers(self.root)
        self.terminal_nodes_count = len(self.terminal_answers)
        self.actions_count = len(self.get_actions_dict())
        self.final_path = None
        if is_recompute_best_path:
            self.final_path = self.get_best_action_path()

    def run(self):
        start_time = time.time()
        final_state = self.run_mcts()
        end_time = time.time()
        self.time_taken = end_time - start_time
        self.update_internal_state()
        return final_state

    def get_time(self):
        return self.time_taken

    def get_terminal_nodes_count(self):
        return self.terminal_nodes_count

    def get_actions_count(self):
        return self.actions_count

    def get_total_rollouts(self):
        return self.total_rollouts

    def get_total_steps(self):
        return self.total_steps

    def get_max_search_depth(self):
        return self.max_search_depth

    def get_terminal_answers_and_values(self):
        terminal_answers, terminal_values = self.collect_terminal_answers(self.root)
        return terminal_answers, terminal_values

    def save_tree(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.root, f)

    def rerank_terminal_and_leaf(self, is_weighted_vote=False):
        terminal_states = [n.get_end_state() for n in self.collect_terminal_nodes(self.root)]
        final_state_global = self.get_final_state_from_global()
        final_state = self.get_final_state_greedy()
        terminal_states.append(final_state_global)
        terminal_states.append(final_state)

        terminal_paths_orm_values = []
        terminal_paths = []
        for terminate_state in terminal_states:
            path = terminate_state.cur_answer
            if self.value_weight > 0 and self.fastrollout_weight == 0:
                # terminal_paths_orm_values.append(self.root.state.get_value(is_additional=False, path=path))
                terminal_paths_orm_values.append(0)
            else:
                terminal_paths_orm_values.append(self.root.state.get_outcome_70b(path))
            terminal_paths.append(path)

        argmax_orm = np.argmax(terminal_paths_orm_values)
        reranked_model_answer = terminal_paths[argmax_orm]
        reranked_state = terminal_states[argmax_orm]
        if not is_weighted_vote:
            return reranked_model_answer, reranked_state
        get_pred_answer_number_list = [state.get_pred_answer_number() for state in terminal_states]
        weighted_vote_answer = weighted_vote(get_pred_answer_number_list, terminal_paths_orm_values)
        weighted_vote_index = get_pred_answer_number_list.index(weighted_vote_answer)
        return reranked_model_answer, reranked_state, terminal_paths[weighted_vote_index], terminal_states[
            weighted_vote_index]

    # Similar to rerank_terminal_and_leaf, but return more infomation. Used in offline analysis.
    def rerank_terminal_and_leaf_all(self):
        terminal_states = [n.get_end_state() for n in self.collect_terminal_nodes(self.root)]
        final_state_global = self.get_final_state_from_global()
        final_state = self.get_final_state_greedy()
        terminal_states.append(final_state_global)
        terminal_states.append(final_state)

        terminal_paths_orm_values = []
        terminal_paths = []
        for terminate_state in terminal_states:
            path = terminate_state.cur_answer
            if self.value_weight > 0 and self.fastrollout_weight == 0:
                # terminal_paths_orm_values.append(self.root.state.get_value(is_additional=False, path=path))
                terminal_paths_orm_values.append(0)
            else:
                terminal_paths_orm_values.append(self.root.state.get_outcome_70b(path))
            terminal_paths.append(path)

        print(f"terminal_paths_orm_values: {terminal_paths_orm_values}")
        return terminal_paths, terminal_states, terminal_paths_orm_values

    @staticmethod
    def update_config_for_tree_nodes(node, new_config, default_config=None):
        """
        Update the config and missing attributes for the given node, its state, and its children.
        """
        if default_config is None:
            default_config = {}

        node.config.update(new_config)
        for key, value in new_config.items():
            setattr(node, key, value)
            setattr(node.state, key, value)

        for key, value in default_config.items():
            if not hasattr(node, key):
                setattr(node, key, value)
            if not hasattr(node.state, key):
                setattr(node.state, key, value)

        for child in node.children:
            MCTS.update_config_for_tree_nodes(child, new_config, default_config)

    @classmethod
    def load_tree(cls, filename, config, default_config=None):
        """
        Load the MCTS tree from a file, and update the config and missing attributes for each node and state.
        """
        with open(filename, 'rb') as f:
            root = pickle.load(f)

        mcts_recover = cls(initial_state=None, config=config)

        MCTS.update_config_for_tree_nodes(root, config, default_config)

        mcts_recover.set_root(root)
        return mcts_recover