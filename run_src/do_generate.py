# Licensed under the MIT license.

import sys
import os, json, time
from tqdm import tqdm
import torch

sys.path.append(".")

from common.utils import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from run_src.mma_utils import compute_flzx,compute_acc, compute_scm,compute_lqa,\
                                compute_jejs,compute_cfm,compute_zmyc,compute_agi,\
                                compute_choice
                                
from MCTS_for_reasoning import Generator, search_for_answers
from transformers import AutoTokenizer

from openai import OpenAI
def main(args):
            
    # use deepseek as general model
    # client = OpenAI(api_key="your_key", base_url="https://api.deepseek.com")
    fix_seeds(args.seed)
    cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1
    base_model_device = f'cuda:{cuda_devices[0]}'
    if len(cuda_devices) > 1:
        domain_model_device = f'cuda:{cuda_devices[1]}'
        print(f'base model on {base_model_device}, domain model on {domain_model_device}')
    else:
        raise ValueError('Please set CUDA_VISIBLE_DEVICES to at least 2 GPUs')
    
    if args.api == 'huggingface':
        from models.HuggingFace_API import load_model
        base_tokenizer, base_model = load_model(args.base_model_ckpt)
        domain_tokenizer, domain_model = load_model(args.domain_model_ckpt,use_causal=False)
    elif args.api == 'vllm':
        from models.vLLM_API import ModelProxy
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_ckpt, trust_remote_code=True)
        base_model = ModelProxy(args.base_model_ckpt, args.seed, args.dtype, device=base_model_device)
        domain_tokenizer = AutoTokenizer.from_pretrained(args.domain_model_ckpt, trust_remote_code=True)
        domain_model = ModelProxy(args.domain_model_ckpt, args.seed, args.dtype, device=domain_model_device)
    # generator = Generator(args, client, domain_tokenizer, domain_model)
    generator = Generator(args, base_tokenizer, base_model, domain_tokenizer, domain_model)

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    output_file = os.path.join("output",args.dataset_name,"output.json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)

    data=[]

    answer_list=[]
    answer_depth_list=[]
    expected_answer_list=[]

    # pre_reward_list=[]

    for _, data_item in enumerate(
        (pbar := tqdm(data_item_list))
    ):
        question, expected_answer = data_item["question"], data_item["answer"]

        answer = search_for_answers(
            args=args, user_question=question, generator=generator
        )

        answer_list.append(answer)
        expected_answer_list.append(expected_answer)
        data.append({
            "question": question,
            "answer": answer,
            "expected_answer": expected_answer
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    if args.dataset_name == "JEJS":
        print("answer:",compute_jejs(answer_list, expected_answer_list))
    elif args.dataset_name == "AGIEVAL":
        print("answer:",compute_agi(answer_list, expected_answer_list))
    else:
        raise ValueError("dataset_name not found")
 
if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")


    # Action1: Propose an one-step thought.
    parser.add_argument("--output_file", type=str, default="output/LAW.json")
    parser.add_argument("--num_a1_steps", type=int, default=None)
    parser.add_argument("--disable_a1", action="store_true")
    # Paraphrasing
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")
    parser.add_argument("--disable_a5", action="store_true")

    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")

    #! -------------------------------------------------------------------------------

    args = parser.parse_args()


    if args.mcts_num_last_votes is None:
        args.mcts_num_last_votes = 5

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)

    args.fewshot_path = os.path.join(prompts_dir, "fewshot", "fewshot.json")
    args.fewshot_rephrased_path = os.path.join(prompts_dir, "fewshot", "fewshot_rephrased.json")

    args.fewshot_context_path = os.path.join(prompts_dir, "fewshot", "fewshot_context.json")
    args.fewshot_context_rephrased_path = os.path.join(prompts_dir, "fewshot", "fewshot_context_rephrased.json")

    args.context_path = os.path.join(prompts_dir, "context", "context.json")
    args.context_rephrased_path = os.path.join(prompts_dir, "context", "context_rephrased.json")

    args.context_answer_path = os.path.join(prompts_dir, "context", "context_answer.json")
    args.context_answer_rephrased_path = os.path.join(prompts_dir, "context", "context_answer_rephrased.json")


    args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.json")


    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)