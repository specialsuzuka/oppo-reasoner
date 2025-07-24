lm_id=deepseek-chat
port=10008
pkill -f -9 "port $port"

python3 tdw-gym/challenge_oppo.py \
--output_dir results_coela_1 \
--lm_id $lm_id \
--experiment_name LMs-$lm_id \
--run_id run_1 \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
--screen_size 256
pkill -f -9 "port $port"