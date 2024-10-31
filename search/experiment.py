# --*-- coding: utf-8 --*--

# Copyright 2023 Tencent
"""MAIN EXPERIMENTS"""

import argparse
from mcts import ProblemState, MCTS
import json
import multiprocessing

import pickle
import os
import numpy as np
import time
import traceback


# for gsm8k
def load_data_gsm8k(file):
    print(f"Loading {file} ...")
    with open(file, "r", encoding='utf-8') as fin:
        data = []
        for line in fin:
            inst = json.loads(line.strip())
            data.append(inst)
    print(f"Loaded total {len(data)} examples.")
    print(data[0])
    return data


def gen_experiment_name(config, suffix):
    experiment_name = (f"{config['exp_name']}" f"{suffix}")
    return experiment_name


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_one_experiment(data, n_attempts=5):
    index, prompt, label, config, args, trees_dir = data

    for attempt in range(n_attempts):
        try:
            initial_state = ProblemState(prompt, label, config, max_lines=config['max_depth'])
            answer_number = initial_state.get_answer_number()

            if args.debug_log_level >= 3:
                print("Starting MCT searching...", flush=True)
            mcts = MCTS(initial_state, config, args=args)
            final_state = mcts.run()
            final_state_global = mcts.get_final_state_from_global()
            _, final_state_rerank, _, final_state_weighted_vote = mcts.rerank_terminal_and_leaf(is_weighted_vote=True)

            if args.debug_log_level >= 3:
                print("Done MCTS.", flush=True)

            pred_answer_greedy = final_state.get_pred_answer_number()
            pred_answer_global = final_state_global.get_pred_answer_number()
            pred_answer_rerank = final_state_rerank.get_pred_answer_number()
            terminal_answers, terminal_values = mcts.get_terminal_answers_and_values()
            max_search_depth = mcts.get_max_search_depth()
            n_rollouts = mcts.get_total_rollouts()
            n_steps = mcts.get_total_steps()
            n_requests = mcts.get_total_requests()

            result = {
                'index': index,
                'answer_number': answer_number,
                'pred_answer_greedy': pred_answer_greedy,
                'pred_answer_global': pred_answer_global,
                'pred_answer_rerank': pred_answer_rerank,
                'terminal_answers': terminal_answers,
                'terminal_values': terminal_values,
                'is_correct': final_state.is_correct(),
                'is_correct_global': final_state_global.is_correct(),
                'is_correct_rerank': final_state_rerank.is_correct(),
                'is_correct_weighted_vote': final_state_weighted_vote.is_correct(),
                'time': mcts.get_time(),
                'best_action_path': mcts.get_best_action_path(),
                'best_action_path_global': mcts.get_best_action_path_global(),
                'actions_dict': mcts.get_actions_dict(),
                'max_search_depth': max_search_depth,
                'n_rollouts': n_rollouts,
                'n_steps': n_steps,
                'n_requests': n_requests,
            }

            if args.save_data:
                tree_filename = os.path.join(trees_dir, f"tree_{index}.p")
                mcts.save_tree(tree_filename)

            return result
        except Exception as e:
            if attempt < n_attempts - 1:
                print(f'Error at sample {index} for attempt {attempt}: {str(e)}', flush=True)
                print(f'Type of error: {type(e).__name__}', flush=True)
                print(f'Traceback: {traceback.format_exc()}', flush=True)
                continue
            else:
                print(f'Error at sample {index} for attempt {attempt}: {str(e)}', flush=True)
                print(f'Type of error: {type(e).__name__}', flush=True)
                print(f'Traceback: {traceback.format_exc()}', flush=True)
                return {}


class SearchExperiment():

    def __init__(self, config, data_path, args):
        self.args = args
        self.config = config
        self.search_every_n = config["search_every_n"]
        self.search_mod_m = config["search_mod_m"]
        self.search_continue = config["search_continue"]
        self.data = load_data_gsm8k(data_path)
        self.prompt_name = 'prompt'
        self.gt_answer_name = 'label'
        self.model_answer_name = 'generated_text'
        self.split_by = '\n'
        self.results = []
        self.start_time = time.time()

        print("Args:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")

    def run_experiment(self):
        n_samples = len(self.data)
        experiment_name = gen_experiment_name(self.config, self.args.suffix)
        trees_dir = f"{self.config['output_dir']}/tree_{experiment_name}"
        if args.save_data:
            os.makedirs(trees_dir, exist_ok=True)

        input_dicts = []
        self.results = list([{
            'index': i,
            'update': False,
        } for i in range(n_samples)])
        self.results_prev_iter = list([{
            'index': i,
            'is_correct': False,
        } for i in range(n_samples)])

        self.n_load_questions = 0
        if self.search_continue:
            self.load_results()

        processed_samples = -1
        for i in range(n_samples):

            if args.search_every_n > 1 and i % args.search_every_n != search_mod_m:
                continue
            if i < args.continue_from:
                continue
            if args.continue_until != -1 and i > args.continue_until:
                break
            if self.search_continue and self.results[i]['update']:
                continue
            if self.args.filter_prev_iter and self.results_prev_iter[i]['is_correct']:
                continue
            
            prompt = self.data[i][self.prompt_name]
            label = self.data[i][self.gt_answer_name]

            try:
                (offline_mean_depth, offline_mean_correct_depth,
                 offline_correct_rate) = self.analyize_offline_data(self.data[i])

                input_dicts.append((i, prompt, label, self.config, self.args, trees_dir))
                self.results[i] = {
                    'index': i,
                    'offline_mean_depth': offline_mean_depth,
                    'offline_mean_correct_depth': offline_mean_correct_depth,
                    'offline_correct_rate': offline_correct_rate,
                    'update': False,
                }
                self.results_prev_iter[i] = {
                    'index': i,
                    'is_correct': False,
                }
            except Exception as e:
                print(f"Error at sample {i}: {str(e)}")
                print(f"Type of error: {type(e).__name__}")
                print(f"Traceback: {traceback.format_exc()}")
                continue

        print(f"Running distributed MCTS on {len(input_dicts)} examples ...")

        with multiprocessing.Pool(processes=args.n_process) as p:

            results = p.imap_unordered(run_one_experiment, input_dicts)
            for result in results:
                if result:
                    index = result.pop('index')
                    self.results[index].update(result)
                    self.results[index]['update'] = True
                    full_result = self.results[index]

                    if full_result['is_correct'] or full_result['is_correct_global'] or full_result['is_correct_rerank'] or full_result['is_correct_weighted_vote']:
                        self.results_prev_iter[index]['is_correct'] = True

                    print(
                        f"Answer number: {full_result['answer_number']}, offline_correct_rate: {full_result['offline_correct_rate']}\n"
                        f"pred_answer_greedy: {full_result['pred_answer_greedy']} {full_result['is_correct']}, {full_result['best_action_path'][-1]}\n"
                        f"pred_answer_global: {full_result['pred_answer_global']} {full_result['is_correct_global']}, {full_result['best_action_path_global'][-1]}\n"
                        f"max_search_depth: {full_result['max_search_depth']}, offline mean_depth: {full_result['offline_mean_depth']}, "
                        f"offline mean_correct_depth: {full_result['offline_mean_correct_depth']}")

                    self.print_result(index)
                    # fix the case when the server is down and then too many writes cause issues with the output file
                    processed_samples += 1
                    if args.save_data and processed_samples % 5 == 0:
                        self.save_results()
                        print(f"Saved {processed_samples} results to {self.config['output_dir']}/exp_{experiment_name}.p.")

        if args.save_data:
            self.save_results()

    def print_result(self, index):
        updated_results = [res for res in self.results if res['update']]

        n_correct = sum(res['is_correct'] for res in updated_results)
        n_correct_global = sum(res['is_correct_global'] for res in updated_results)
        if 'is_correct_rerank' not in updated_results[0]:
            n_correct_rerank = 0
        else:
            n_correct_rerank = sum(res['is_correct_rerank'] for res in updated_results)
        if 'is_correct_weighted_vote' not in updated_results[0]:
            n_correct_weighted_vote = 0
        else:
            n_correct_weighted_vote = sum(res['is_correct_weighted_vote'] for res in updated_results)
        total_time = sum(res['time'] for res in updated_results)
        total_rollouts = sum(res['n_rollouts'] for res in updated_results)
        total_steps = sum(res['n_steps'] for res in updated_results)
        total_requests = sum(res['n_requests'] for res in updated_results)
        n_samples = len(updated_results)
        n_processesed_samples = n_samples - self.n_load_questions

        n_searched = sum(1 for res in updated_results if len(res['terminal_answers']) > 0)
        n_true_in_terminal = sum(1 for res in updated_results if res['answer_number'] in res['terminal_answers'])
        n_incorrect_positive_value = 0
        n_incorrect_negative_value = 0
        for res in updated_results:
            if not res['is_correct'] and res['terminal_values']:
                if res['pred_answer_greedy'] in res['terminal_answers']:
                    greedy_answer_index = res['terminal_answers'].index(res['pred_answer_greedy'])
                    greedy_answer_value = res['terminal_values'][greedy_answer_index]
                    if greedy_answer_value > 0:
                        n_incorrect_positive_value += 1
                    else:
                        n_incorrect_negative_value += 1

        avg_max_search_depth = sum(res['max_search_depth'] for res in updated_results) / n_samples
        avg_offline_mean_depth = sum(res['offline_mean_depth'] for res in updated_results) / n_samples
        avg_offline_correct_rate = sum(res['offline_correct_rate'] for res in updated_results) / n_samples

        wrong_but_positive_value = n_incorrect_positive_value / (n_incorrect_positive_value +
                                                                 n_incorrect_negative_value + 1e-7)
        current_time = time.time()
        if n_processesed_samples > 0:
            avg_time_per_sample = (current_time - self.start_time) / n_processesed_samples
        else:
            avg_time_per_sample = -1

        print(
            f'Q: {index:04d}, N: {n_samples}, Avg acc: {n_correct / n_samples:.3f}, '
            f'Avg acc global: {n_correct_global / n_samples:.3f}, Avg acc rerank: {n_correct_rerank / n_samples:.3f}, Avg acc wv: {n_correct_weighted_vote / n_samples:.3f},  Avg time: {avg_time_per_sample:.2f}s, Avg rollouts: {total_rollouts / n_samples:.1f}, '
            f'Avg steps: {total_steps / n_samples:.1f}, Avg requests: {total_requests / n_samples:.1f}, '
            f'Terminal rate: {n_searched / n_samples:.3f}, '
            f'Answer searched rate: {n_true_in_terminal / (n_searched + 1e-7):.3f}, '
            f'Max depth rate: {avg_max_search_depth / avg_offline_mean_depth:.2f} \n'
            f'Incorrect answer with value > 0: {wrong_but_positive_value:.2f}, '
            f'Avg offline correct rate: {avg_offline_correct_rate:.2f}\n'
            f'-----------------------------')

    def save_results(self):
        experiment_name = gen_experiment_name(self.config, self.args.suffix)
        filename = f"{self.config['output_dir']}/exp_{experiment_name}.p"
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)

        if self.args.filter_prev_iter:
            filename = self.args.prev_iter_file
            with open(filename, 'wb') as f:
                pickle.dump(self.results_prev_iter, f)

    def load_results(self):
        experiment_name = gen_experiment_name(self.config, self.args.suffix)
        filename = f"{self.config['output_dir']}/exp_{experiment_name}.p"
        print(f"Loading {filename} ...")
        try:
            # if 1:
            with open(filename, 'rb') as f:
                results = pickle.load(f)
                # Convert old results to new format
                for idx, res in enumerate(results):
                    if 'index' not in res:
                        # In the old results format, we didn't record the question id.
                        # Here, we assume that
                        # 1. there are no missing questions;
                        # 2. search_every_n == 1
                        res['index'] = idx
                    if 'update' not in res:
                        res['update'] = True
                    self.results[idx] = res
            self.n_load_questions = sum(1 for res in self.results if res['update'])
            last_updated_index = max(idx for idx, res in enumerate(self.results) if res['update'])
            print('-' * 20)
            print(f"Loaded {self.n_load_questions} results from {filename}.")
            self.print_result(last_updated_index)
        except Exception as e:
            print(f"Cannot load: {filename} for {e}. Decoding from beginning.")

        if self.args.filter_prev_iter:
            filename = self.args.prev_iter_file
            print(f"Loading {filename} ...")
            try:
                with open(filename, 'rb') as f:
                    results = pickle.load(f)
                    # Convert old results to new format
                    for idx, res in enumerate(results):
                        self.results_prev_iter[idx] = res
                n_prev_correct = sum(1 for res in self.results_prev_iter if res['is_correct'])
                print(f"N of correct questions from previous round if {n_prev_correct}.")
            except Exception as e:
                print(f"Cannot load prev iter file: {filename} for {e}.")


    def analyize_offline_data(self, inst):
        correct_answer = ProblemState.extract_answer(inst[self.gt_answer_name],
                                                     self.config['dataset'],
                                                     self.config['policy_type'],
                                                     is_reference=True)
        # return dummy values for OOM data (e.g. math_qa)
        if self.model_answer_name is None or self.model_answer_name not in inst:
            return 1, 1, 1
        generated_texts = inst[self.model_answer_name]
        correct_list = []
        step_list = []
        for i in range(len(generated_texts)):
            generated_text = generated_texts[i]
            candidate_answer = ProblemState.extract_answer(generated_text,
                                                           self.config['dataset'],
                                                           self.config['policy_type'],
                                                           is_reference=False)
            generated_text_step = generated_text.split(self.split_by)
            n_steps = len(generated_text_step)
            correct_list.append(candidate_answer == correct_answer)
            step_list.append(n_steps)

        mean_depth = sum(step_list) / len(step_list)
        correct_rate = sum(correct_list) / len(correct_list)
        # mean_correct_depth, 0 if no correct answers
        if correct_rate == 0:
            mean_correct_depth = 0
        else:
            mean_correct_depth = np.mean([step_list[i] for i in range(len(step_list)) if correct_list[i]])
        return mean_depth, mean_correct_depth, correct_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default=
        '/apdcephfs/share_300000800/user/baolinpeng/local_exp/codes/vllm-inference/llama-70b-gsm8k_cot_8_shot_test.decoded.n50.topp1.temp0.8.jsonl',
        help='Path to the data file')
    parser.add_argument('--dataset', type=str, default="gsm8k", help='Dataset name')
    parser.add_argument('--search_every_n', type=int, default=1, help='Seach one example for every n instances.')
    parser.add_argument('--search_mod_m', type=int, default=0, help='Mod m for evaluation.')
    parser.add_argument('--search_continue', type=str2bool, default=False, help='Continue stopped search.')
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth of the MCTS')
    parser.add_argument('--n_total_children', type=int, default=3, help='Number of total children in the MCTS')
    parser.add_argument('--init_children',
                        type=int,
                        default=10,
                        help='Initial number of children to expand at the root node')
    parser.add_argument('--max_split_depth', type=int, default=1, help='Maximum depth to split the tree')
    parser.add_argument('--c_param', type=float, default=1.0, help='Exploration parameter for the UCB formula')
    parser.add_argument('--search_time', type=int, default=200, help='Number of iterations for the MCTS')
    parser.add_argument('--min_terminals', type=int, default=3, help='Minimum number of terminals for the MCTS')
    parser.add_argument('--save_data', type=str2bool, default=True, help='Whether to save the data_to_save')
    parser.add_argument('--state_merge_method', type=str, default="edit_dist", help='Method to merge the states')
    parser.add_argument('--max_n_repeat', type=int, default=1, help='Maximum number of repeat actions in the MCTS')
    parser.add_argument('--edit_dist_thres', type=int, default=50, help='Edit distance threshold for state merge')
    parser.add_argument('--similarity_thres', type=float, default=0.4, help='Similarity threshold for state merge')
    parser.add_argument('--width_increase_factor', type=int, default=0, help='Width increase factor for the MCTS')
    parser.add_argument('--suffix', type=str, default='', help='Experiment suffix to add to the output file')
    parser.add_argument('--policy_url', type=str, default="http://localhost:8000/v1", help='URL for the policy model')
    parser.add_argument('--value_url',
                        type=str,
                        default="http://localhost:8001/predict",
                        help='URL for the value model')
    parser.add_argument('--policy_type', type=str, default="llama2_70b", help='Type of the policy model')
    parser.add_argument('--value_type', type=str, default="llama2_13b", help='Type of the value model')
    parser.add_argument('--policy_model',
                        type=str,
                        default="/apdcephfs/share_300000800/user/dyu/share/model/pretrain/llama-2/hf/Llama-2-70b-hf",
                        help='Path to the policy model')
    parser.add_argument('--simulate_policy_url',
                        type=str,
                        default="http://localhost:8000/v1",
                        help='URL for the simulate policy model')
    parser.add_argument('--simulate_policy_type',
                        type=str,
                        default="llama2_70b",
                        help='Type of the simulate policy model')
    parser.add_argument('--simulate_policy_model',
                        type=str,
                        default="/apdcephfs/share_300000800/user/dyu/share/model/pretrain/llama-2/hf/Llama-2-70b-hf",
                        help='Path to the simulate policy model')
    parser.add_argument('--step_reward_url',
                        type=str,
                        default="http://localhost:8100/predict",
                        help='URL for the step reward model')
    parser.add_argument('--step_reward_type', type=str, default="llama2_13b", help='Type of the step reward model')
    parser.add_argument('--step_reward_weight', type=float, default=0.0, help='Weight for the step reward')
    parser.add_argument('--step_reward_mode', type=str, default="value", help='Method to use the step reward')
    parser.add_argument('--state_merge_url',
                        type=str,
                        default="http://localhost:8200/predict",
                        help='URL for the state merge model')
    parser.add_argument(
        '--state_merge_model',
        type=str,
        default="/apdcephfs/share_300000800/user/dyu/share/model/pretrain/llama-2/hf/Llama-2-70b-chat-hf",
        help='Path to the state merge model')
    parser.add_argument('--state_merge_model_type',
                        type=str,
                        default="llama2_70b_chat",
                        help='Type of the state merge model')
    parser.add_argument('--stop_repeat_node_thres',
                        type=int,
                        default=0,
                        help='Stop expanding a node if it has more than this number of repeat actions')
    parser.add_argument('--n_simulations', type=int, default=0, help='Number of simulations with fast rollout')
    parser.add_argument('--fastrollout_weight', type=float, default=0.0, help='Weight for the fast rollout')
    parser.add_argument('--orm_url', type=str, default="http://localhost:8103/generate", help='URL for the orm model')
    parser.add_argument('--orm_type', type=str, default="llama2_70b_lm", help='Type of the orm model')
    parser.add_argument('--orm_use_sympy', type=str2bool, default=False, help='Whether to include sympy for orm')
    parser.add_argument(
        '--orm_preset_code_path',
        type=str,
        default='/apdcephfs/share_300000800/user/baolinpeng/exp.tencent_chat/data/math_search/MATH_preset_code.json',
        help='Path to preset code and code output')
    parser.add_argument('--orm_calibrate_logits',
                        type=str2bool,
                        default=False,
                        help='Whether to calibrate with softmax')
    parser.add_argument('--value_weight', type=float, default=0.0, help='Weight for the value')
    parser.add_argument('--add_value_weight',
                        type=float,
                        default=0.0,
                        help='Additional weight for the value with a different type')
    parser.add_argument('--add_value_type', type=str, default="llama2_13b", help='Type of the additional value')
    parser.add_argument('--add_value_url',
                        type=str,
                        default="http://localhost:8001/predict",
                        help='URL for the additional value model')
    parser.add_argument('--consistency_url',
                        type=str,
                        default="http://30.159.161.95:8322/v1",
                        help='URL for the consistency model')
    parser.add_argument(
        '--consistency_model',
        type=str,
        default=
        "/apdcephfs/share_300000800/user/baolinpeng/local_exp/codes/lit-gpt-2/outputs/jp_consistency_v2_llama13b_v2_finetuned_epoch3",
        help='Path for the consistency model')
    parser.add_argument('--greedy_path', type=str2bool, default=False, help='Whether to use the greedy path')
    parser.add_argument('--use_calculator', type=str2bool, default=False, help='Whether to use the calculator')
    parser.add_argument('--use_math_extractor', type=str2bool, default=False, help='Whether to use the math extractor')
    parser.add_argument('--math_extractor_url',
                        type=str,
                        default="http://localhost:8000/generate",
                        help='URL for the math extractor model')
    parser.add_argument('--use_rule_based_step_extend',
                        type=str2bool,
                        default=False,
                        help='Whether to use the rule based step extend')
    parser.add_argument('--stop_expansion_rollout_var',
                        type=float,
                        default=0.0,
                        help='Stop expansion if the rollout variance is less than this value')
    parser.add_argument('--use_consistency_as_orm',
                        type=str2bool,
                        default=False,
                        help='Whether to use the consistency as orm')
    parser.add_argument('--use_match_as_orm',
                        type=str2bool,
                        default=False,
                        help='Whether to use answer matching as orm')
    parser.add_argument('--exec_code', type=str2bool, default=False, help='Whether to execute the code')
    parser.add_argument('--exp_name', type=str, default="mcts", help='Experiment name')
    parser.add_argument('--output-dir', type=str, default="data/trees", help='Output path')
    parser.add_argument('--debug-log-level',
                        type=int,
                        default=0,
                        choices=range(0, 6),
                        help='Granularity level to debug info. ')
    parser.add_argument('--n_process',
                        type=int,
                        default=1,
                        help='Number of processes to run the experiment in parallel')
    parser.add_argument('--high_value_rollout',
                        type=str2bool,
                        default=False,
                        help='Rollout to terminal if norm_value score >= 0.9')
    parser.add_argument('--low_value_prune',
                        type=str2bool,
                        default=False,
                        help='Prune the node if norm_value score <= 0.1')
    parser.add_argument('--continue_from', type=int, default=-1, help='Run the experiment from')
    parser.add_argument('--continue_until', type=int, default=-1, help='Run the experiment until')
    parser.add_argument('--filter_prev_iter', type=str2bool, default=False, help='Filter the previous iteration')
    parser.add_argument('--prev_iter_file', type=str, default='/apdcephfs/share_300000800/user/yaptian/exp.tencent_chat/math-search/prev_iter_files/prev_iter.p',help='Previous iteration file')
    args = parser.parse_args()
    if args.dataset == "metamath":
        args.dataset = "gsm8k"

    search_mod_m = args.search_mod_m % args.search_every_n
    config = {
        "search_every_n": args.search_every_n,
        "search_mod_m": args.search_mod_m,
        "search_continue": args.search_continue,
        "max_depth": args.max_depth,
        "n_total_children": args.n_total_children,
        "init_children": args.init_children,
        "max_split_depth": args.max_split_depth,
        "c_param": args.c_param,
        "search_time": args.search_time,
        "state_merge_method": args.state_merge_method,
        "max_n_repeat": args.max_n_repeat,
        "edit_dist_thres": args.edit_dist_thres,
        "similarity_thres": args.similarity_thres,
        "width_increase_factor": args.width_increase_factor,
        "policy_url": args.policy_url,
        "value_url": args.value_url,
        "simulate_policy_url": args.simulate_policy_url,
        "state_merge_url": args.state_merge_url,
        "policy_model": args.policy_model,
        "simulate_policy_model": args.simulate_policy_model,
        "state_merge_model": args.state_merge_model,
        "policy_type": args.policy_type,
        "value_type": args.value_type,
        "simulate_policy_type": args.simulate_policy_type,
        "n_simulations": args.n_simulations,
        "step_reward_url": args.step_reward_url,
        "step_reward_type": args.step_reward_type,
        "step_reward_weight": args.step_reward_weight,
        "step_reward_mode": args.step_reward_mode,
        "state_merge_model_type": args.state_merge_model_type,
        "stop_repeat_node_thres": args.stop_repeat_node_thres,
        "fastrollout_weight": args.fastrollout_weight,
        "orm_url": args.orm_url,
        "orm_type": args.orm_type,
        "orm_use_sympy": args.orm_use_sympy,
        "orm_preset_code_path": args.orm_preset_code_path,
        "orm_calibrate_logits": args.orm_calibrate_logits,
        "value_weight": args.value_weight,
        "add_value_weight": args.add_value_weight,
        "add_value_type": args.add_value_type,
        "add_value_url": args.add_value_url,
        "greedy_path": args.greedy_path,
        "min_terminals": args.min_terminals,
        "use_calculator": args.use_calculator,
        "use_math_extractor": args.use_math_extractor,
        "math_extractor_url": args.math_extractor_url,
        "consistency_url": args.consistency_url,
        "consistency_model": args.consistency_model,
        "use_rule_based_step_extend": args.use_rule_based_step_extend,
        "stop_expansion_rollout_var": args.stop_expansion_rollout_var,
        "use_consistency_as_orm": args.use_consistency_as_orm,
        "use_match_as_orm": args.use_match_as_orm,
        "exec_code": args.exec_code,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "high_value_rollout": args.high_value_rollout,
        "low_value_prune": args.low_value_prune,
    }

    experiment = SearchExperiment(config, args.data_path, args)
    experiment.run_experiment()
