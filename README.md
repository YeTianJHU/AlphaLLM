# AlphaLLM

AlphaLLM combines Monte Carlo Tree Search (MCTS) with Large Language Models (LLMs) to establish a self-improving loop for complex reasoning and planning tasks.

## Setup

### Starting Services

1. **Policy Service**
   - Uses Llama models (Llama-2-70B)
   - Start multiple policy servers for parallel processing
   ```bash
   # Example policy server startup
   python server/policy_server.py --host 0.0.0.0 --port 8002 --model /path/to/llama/model --tensor-parallel-size 4
   ```

2. **Value Service**
   - Evaluates the quality of current solution paths
   - Multiple servers can be started on different ports
   ```bash
   python server/value_server.py --host 0.0.0.0 --port 8006 --model /path/to/value/model
   ```

3. **Outcome Reward Model (ORM) Service**
   ```bash
   # Start using the step reward server script with ORM configuration
   CUDA_VISIBLE_DEVICES=$gpus python3 server/step_server_70b.py \
     --host 0.0.0.0 \
     --port $ORM_PORT \
     --model $ORM_MODEL_PATH \
     --tensor-parallel-size $tp \
     >& vllm_log_orm &
   ```

4. **Process Reward Model (PRM) Service**
   ```bash
   # Start using the step reward server script with PRM configuration
   python3 server/step_server_70b.py \
     --host 0.0.0.0 \
     --port $PRM_PORT \
     --model $PRM_MODEL_PATH \
     --tensor-parallel-size $tp \
     >& vllm_log_step &
   ```

### Running MCTS

The main MCTS search can be started using the `run_mcts.sh` script. Key parameters include:

- `SEARCH_TIME`: Total number of search iterations (default: 40)
- `MIN_TERMINALS`: Minimum number of terminal nodes to explore (default: 40) 
- `INIT_CHILDREN`: Initial number of children to expand at root node (default: 40)
- `N_TOTAL_CHILDREN`: Number of children to expand at each node (default: 3)
- `C_PARAM`: Exploration parameter for UCT (default: 1)
- `WIDTH_INCREASE_FACTOR`: Factor to increase tree width in adaptive child allocation (default: 2)
- `MAX_SPLIT_DEPTH`: Maximum depth for tree splitting (default: 10)
- `MAX_DEPTH`: Maximum search depth (default: 10)

Additional configuration options:

- `VALUE_WEIGHT`: Weight of value network in total reward (0-1)
- `N_SIMULATIONS`: Number of simulations with simulation policy model
- `FASTROLLOUT_WEIGHT`: Weight of fast rollout reward (0-1)
- `PROCESS_REWARD_WEIGHT`: Weight of process reward in total reward (0-1)
- `USE_CALCULATOR`: Enable calculator for evaluating mathematical expressions
- `USE_MATH_EXTRACTOR`: Enable math expression extraction
- `EXEC_CODE`: Enable code execution for verification
- `SAVE_DATA`: Save search trees and results

Example usage:

```bash
./scripts/run_mcts.sh 
```