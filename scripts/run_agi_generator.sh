CUDA_VISIBLE_DEVICES=0,1 python run_src/do_generate.py \
    --dataset_name AGIEVAL \
    --test_json_filename test_one \
    --base_model_ckpt your_path/Llama_8B_Instruct \
    --domain_model_ckpt your_path/LawLLM \
    --base_model_prompt_type llama3.1 \
    --domain_model_prompt_type baichuan \
    --max_tokens 4096 \
    --temperature 0.2 \
    --top_p 0.7 \
    --top_k 15 \
    --note default \
    --num_rollouts 15 \
    --dtype bfloat16
    # --verbose