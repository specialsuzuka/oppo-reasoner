lm_id=deepseek-chat
port=10004
pkill -f -9 "port $port"

python3 tdw-gym/challenge_oppo.py \
--output_dir results_oppo \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--port $port \
--agents lm_agent_oppo lm_agent_oppo \
--communication \
--prompt_template_path LLM/zmwang_prompt_v3.csv \
--max_tokens 512 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 1 2 3 4 \
--screen_size 256

pkill -f -9 "port $port"