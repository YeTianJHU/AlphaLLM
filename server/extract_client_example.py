import requests
import re
import json
import sympy


operators = ["+", "-", "*", "/", "="]


def remove_enclosed_strings(text):
    # Replace any string enclosed in << and >> with an empty string
    return re.sub(r'<<.*?>>', '', text)


def replace_enclosed_strings(text):
    # Replace any string enclosed in << >> with [[ ]]
    return re.sub(r'<<([^>]*)>>', r'[[\1]]', text)


def identify_and_format_decimal(input_string):
    try:
        float_number = float(input_string)
        return "{:.2f}".format(float_number)
    except ValueError:
        return input_string


def get_cal(step_cal_url, step, value_gsm=False):
    """
    Request the vllm server for 7b model
    """

    # filter those plain-text or empty steps
    if not any(char in step for char in operators) or step.startswith("####") or len(step.strip()) == 0:
        return step

    clean_step = remove_enclosed_strings(step).strip()
    # "2 + 2/2 = 3 bolts."

    # extract code from the given step
    DEBUG= False
    payload = {"prompt": clean_step, "max_tokens": 4096, "logprobs": 10, "temperature": 0, "stop": ['</s>']}

    if DEBUG:
        print("get_cal_step_7b", payload)
    response = requests.post(step_cal_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        pred = result["text"][0].strip()
    else:
        print("Error:", response.status_code, response.text)
        pred = ""

    if not pred:
        return step

    # add executed results of code into the step
    new_pred = []
    if pred:
        try:
            exp_list = json.loads(pred)
        except:
            exp_list = []
        if not isinstance(exp_list, list):
            exp_list = []

        for exp in exp_list:
            if exp.count("**") > 1 or len(exp) > 300:
                continue
            if "sympy." in exp and ("**" in exp or "^" in exp):
                continue

            try:
                result = eval(exp)
                result = identify_and_format_decimal(str(result))
                new_pred.append(exp + "->" + result)
            except:
                new_pred.append(exp)

    if value_gsm:
        step = replace_enclosed_strings(step)

    if len(new_pred):
        # return clean_step + " " + json.dumps(new_pred)
        # enclosed version
        return step + " " + json.dumps(new_pred)
    else:
        # return clean_step
        # enclosed version
        return step



if __name__ == '__main__':

    step_cal_url = "http://9.206.62.242:8000/generate"
    # step_cal_url = "http://localhost:8000/generate"

    input_text = "Melanie sold 1/3 + 2 + 1/2 of what she started with"
    # input_text = "The vertical asymptote(s) occur(s) at the value of $x$ that makes the denominator of the function $y=\\frac{2}{x^2+x-6}$ equal to zero.\n\n$x^2+x-6=0$\n\nThis quadratic can be factored as $(x-3)(x+2)=0$."
    # input_text = "That means she'll have to pay $1300 + $130 = <<1300+130=1430>>1430 total."

    output = get_cal(step_cal_url, input_text)
    # That means she'll have to pay $1300 + $130 = <<1300+130=1430>>1430 total. ["eval('1300+130')->1430.00"]
    print(output)
    # value_gsm: use [[ ]] to replace << >> following the format of the current val training data; use False (default) for value_math, step_gsm/math
    output_1 = get_cal(step_cal_url, input_text, True)
    # That means she'll have to pay $1300 + $130 = [[1300+130=1430]]1430 total. ["eval('1300+130')->1430.00"]
    print(output_1)


    # current server version
    # [1024] value_net_gsm8k_math_llama2_13b_lr1e-6_maxseq1024_nodes4_devicebatch8_nsamples2_calv2both_space_rubric_search
    # gsm: 0.8320410865588476
    # math: 0.8902440244524708

    # new valuenet paths
    # [2048] value_net_gsm8k_math_llama2_13b_lr1e-6_maxseq2048_nodes4_devicebatch2_nsamples2_calv6both_gsmenclosed_space_rubric
    # gsm:  0.8410804356644802
    # math: 0.892471827298589

    # [1024] value_net_gsm8k_math_llama2_13b_lr1e-6_maxseq1024_nodes4_devicebatch8_nsamples2_calv6both_gsmenclosed_space_rubric
    # gsm:  0.8359139998928443
    # math: 0.8942293903223978