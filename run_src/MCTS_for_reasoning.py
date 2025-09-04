# Licensed under the MIT license.

import sys
sys.path.append(".")

import numpy as np, os, random, json, math, wandb
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy
import re
import random
import importlib
import logging
from models.IO_System import IO_System
from common.utils import  read_json
# from eval_src.Evaluator import Evaluator
from collections import Counter
from MCTS_backbone import MCTS_Searcher, MCTS_Node
from run_src.mma_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_context,
    compute_flzx,compute_acc,compute_scm,compute_lqa,compute_jejs,compute_cfm,\
    extract_jejs,extract_choice,extract_zmyc,extract_agi,compute_agi,compute_choice,\
)


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, base_tokenizer, base_model, domain_tokenizer, domain_model) -> None:
        self.io_base = IO_System(args, base_tokenizer, base_model)
        self.io_domain = IO_System(args, domain_tokenizer, domain_model)

        self.base_model_prompt_type = args.base_model_prompt_type
        self.domain_model_prompt_type = args.domain_model_prompt_type

        # Number of answer candidates generated per question
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.dataset_name = args.dataset_name

        # The number of final votes used in the MCTS algorithm
        self.mcts_num_last_votes = args.mcts_num_last_votes

        self.fewshot_cot = read_json(args.fewshot_path)
        self.fewshot_cot_rephrased = read_json(args.fewshot_rephrased_path)

        self.fewshot_cot_context = read_json(args.fewshot_context_path)
        self.fewshot_cot_context_rephrased = read_json(args.fewshot_context_rephrased_path)

        self.context_cot = read_json(args.context_path)
        self.context_cot_rephrased = read_json(args.context_rephrased_path)

        self.context_cot_answer = read_json(args.context_answer_path)
        self.context_cot_answer_rephrased = read_json(args.context_answer_rephrased_path)

        self.rephrasing_prompt_template = read_json(args.rephrasing_prompt_template_path)

        self._load_dataset_specific_prompts()

    def _load_dataset_specific_prompts(self):
        dataset_map = {
            "LEGAL_SCM": "prompts.LEGAL_SCM.prompt_template",
            "LEGAL_LQA": "prompts.LEGAL_LQA.prompt_template",
            "JEJS": "prompts.JEJS.prompt_template",
            "LEGAL_CFM": "prompts.LEGAL_CFM.prompt_template",
            "ZMYC": "prompts.ZMYC.prompt_template",
            "AGIEVAL": "prompts.AGIEVAL.prompt_template",
            "FINEVAL": "prompts.FINEVAL.prompt_template",
            "KNOWLEDGE": "prompts.KNOWLEDGE.prompt_template",
            "CALUATION": "prompts.CALUATION.prompt_template",
        }
        
        try:
            module_name = dataset_map[self.dataset_name]
        except KeyError:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        module = importlib.import_module(module_name)

        fewshot_prompt = getattr(module, "fewshot_prompt")
        fewshot_context_prompt = getattr(module, "fewshot_context_prompt")
        context_prompt = getattr(module, "context_prompt")
        context_answer_prompt = getattr(module, "context_answer_prompt")
        question_parahrased_prompt = getattr(module, "question_parahrased_prompt")


        self.generate_fewshot_prompt = fewshot_prompt
        self.generate_fewshot_context_prompt = fewshot_context_prompt
        self.generate_context_prompt = context_prompt
        self.generate_context_answer_prompt = context_answer_prompt
        self.generate_question_parahrased_prompt = question_parahrased_prompt
  
    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            pred_list=[]
            if self.dataset_name == "JEJS":
                for io_output in io_output_list:
                    idx = io_output.rfind("最终金额")
                    if idx != -1:
                        pred = io_output[idx:]
                        pred_list.append(pred)
                    else:
                        pred_list.append('0.00')
                pred_list, ref_list = extract_jejs(pred_list,['' for i in range(len(io_output_list))])
            elif self.dataset_name == "LEGAL_LQA" or self.dataset_name == "KNOWLEDGE" or self.dataset_name == "CALUATION" or self.dataset_name == "FINEVAL":
                for io_output in io_output_list:
                    idx = io_output.rfind("正确答案")
                    if idx != -1:
                        pred = io_output[idx:]
                        pred_list.append(pred)
                    else:
                        pred_list.append('P')
                pred_list, ref_list = extract_choice(io_output_list,['' for i in range(len(io_output_list))])
            elif self.dataset_name == "ZMYC":
                for io_output in io_output_list:
                    idx = io_output.rfind("罪名")
                    if idx != -1:
                        pred = io_output[idx:]
                        pred_list.append(pred)
                    else:
                        pred_list.append('无')
                pred_list, ref_list = extract_zmyc(io_output_list,['' for i in range(len(io_output_list))])
            elif self.dataset_name == "AGIEVAL":
                for io_output in io_output_list:
                    idx = io_output.rfind("最终答案")
                    if idx != -1:
                        pred = io_output[idx:]
                        pred_list.append(pred)
                    else:
                        pred_list.append('无')
                pred_list, ref_list = extract_agi(io_output_list,['' for i in range(len(io_output_list))])
            else:
                raise ValueError("dataset_name unknown")
            pred_list = [' '.join(pred) if isinstance(pred, list) else pred for pred in pred_list]
            answer_counter = Counter(pred_list)
            most_common_answer, most_common_count = answer_counter.most_common(1)[0]
            if most_common_answer == '0.0' or most_common_answer == '0.00' or most_common_answer == 'P' or most_common_answer == '无':
                for i, pred in enumerate(pred_list):
                    if pred != '0.0' and pred != '0.00' and pred != 'P' and pred != '无':
                        most_confident_answer_full_completion = io_output_list[i]
                        confidence =  answer_counter.most_common(1)[0][1] / len(pred_list)# You can modify the confidence calculation here if needed
                        return most_confident_answer_full_completion, confidence
            most_confident_answer_full_completion = io_output_list[pred_list.index(most_common_answer)]
            confidence = most_common_count / len(pred_list)
        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, model_type:str, question: str, context: str, paraphrased: bool, num_return: int, verbose: bool):
        fewshot_cot = self.fewshot_cot if not paraphrased else self.fewshot_cot_rephrased
        fewshot_cot_context = self.fewshot_cot_context if not paraphrased else self.fewshot_cot_context_rephrased
        prompt_type = self.base_model_prompt_type if model_type == "base_model" else self.domain_model_prompt_type
        if context is None or context == "":
            io_input = self.generate_fewshot_prompt(
                question=question,
                few_shot_prompt=fewshot_cot,
                paraphrased=paraphrased,
                prompt_type=prompt_type
            )
        else:
            io_input = self.generate_fewshot_context_prompt(
                question=question,
                context=context,
                few_shot_prompt=fewshot_cot_context,
                paraphrased=paraphrased,
                prompt_type=prompt_type
            )
        

        if model_type == "base_model":
            verbose_print(f"model_type:base,io_input: {io_input}",verbose)
            io_output_list = self.io_base.generate(
                io_input,
                num_return=num_return,
                max_tokens=self.max_tokens,
            )
        elif model_type == "domain_model":
            verbose_print(f"model_type:domain,io_input: {io_input}",verbose)
            io_output_list = self.io_domain.generate(
                io_input,
                num_return=num_return,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError("model_type must be 'base_model' or 'domain_model'")
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list
    
    def _context_answer_reward(self, model_type:str, question: str, context: str, part_answer:str, verbose: bool):
        if "条件" in question:
            pa = True
        else:
            pa = False
        fewshot_cot = self.fewshot_cot if not pa else self.fewshot_cot_rephrased
        fewshot_cot_context = self.fewshot_cot_context if not pa else self.fewshot_cot_context_rephrased
        prompt_type = self.base_model_prompt_type if model_type == "base_model" else self.domain_model_prompt_type
        if context is None or context == "":
            io_input = self.generate_fewshot_prompt(
                question=question,
                few_shot_prompt=fewshot_cot,
                paraphrased=pa,
                prompt_type=prompt_type
            )
        else:
            io_input = self.generate_fewshot_context_prompt(
                question=question,
                context=context,
                few_shot_prompt=fewshot_cot_context,
                paraphrased=pa,
                prompt_type=prompt_type
            )
        io_input_list=[]
        io_input += "\n"
        for p_a in part_answer:
            p_a = p_a.replace("\n", "")
            io_in = io_input + p_a
            io_input_list.append(io_in)
        verbose_print(f"When calculating rewards, the input is：{io_input}",verbose)
        if model_type == "base_model":
            io_output_list = self.io_base.generate(
                io_input_list,
                num_return=3,
                max_tokens=self.max_tokens,
            )
        elif model_type == "domain_model":
            io_output_list = self.io_domain.generate(
                io_input_list,
                num_return=3,
                max_tokens=self.max_tokens,
            )
        elif model_type == "reward_model_llama" or model_type == "reward_model_phi":
            io_output_list = self.io_reward.generate(
                io_input_list,
                num_return=3,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError("model_type must be 'base_model' or 'domain_model'")
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    def _context_cot_question(self, question: str, context:str, answer: str, paraphrased: bool, verbose: bool):
        context_cot_prompt = self.context_cot if not paraphrased else self.context_cot_rephrased
        context_cot_answer_prompt = self.context_cot_answer  if not paraphrased else self.context_cot_answer_rephrased
        if context is None:
           context = "" 
        if context == "":
            io_input = self.generate_context_prompt(question=question, context_cot_prompt=context_cot_prompt, paraphrased=paraphrased, prompt_type=self.domain_model_prompt_type)
        else:
            io_input = self.generate_context_answer_prompt(question=question, context_cot_prompt=context_cot_answer_prompt, context=context, answer=answer, prompt_type=self.domain_model_prompt_type)
        verbose_print(f"============Generate knowledge based on the following content：{io_input}",verbose)
        io_output_list = self.io_domain.generate(
            io_input,
            num_return=1,
            max_tokens=self.max_tokens,
        )
        generated_answer = io_output_list[0].strip()
        if context =='':
            re_context = generated_answer
        elif "[更正]" in generated_answer:
            index_of_keyword = generated_answer.find("[更正]")
            re_context = generated_answer[index_of_keyword + len("[更正]"):].strip()
        elif "[无需更改]" in generated_answer:
            re_context = context
        elif "[Correction]" in generated_answer:
            index_of_keyword = generated_answer.find("[Correction]")
            re_context = generated_answer[index_of_keyword + len("[Correction]"):].strip()
        elif "[No change required]" in generated_answer:
            re_context = context
        else:
            re_context = context
        
        return io_input, re_context


    def generate_answers(self, model_type: str, user_question: str, context:str, paraphrased: bool, verbose: bool):
        answer_list, value_list = [], []

        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            model_type=model_type, question=user_question, context=context, paraphrased=paraphrased, num_return=num_return, verbose=verbose
        )
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return answer_list, value_list

    def generate_context(self, user_question: str, context:str, answer: str, paraphrased: bool, verbose:bool):
        io_input, re_context = self._context_cot_question(question=user_question, context=context, answer=answer, paraphrased=paraphrased, verbose=verbose)
        verbose_print(f"---- Generate contextual knowledge：\n{re_context}\n",verbose) if context is None or context =='' else  verbose_print(f"---- 审查结果：\n{re_context}\n",verbose)
        answer,value = self.generate_answers(model_type="base_model", user_question=user_question, context=re_context, paraphrased=paraphrased, verbose=verbose)
        verbose_print(f"---- Based on contextual knowledge, generate the answer using the base_model.：\n{answer[0]}\n",verbose)

        return answer, re_context,value

    def generate_reward(self, model_type:str, user_question: str, context:str, answer: str, verbose:bool, value:float,node_type:str):
        input_answer = []
        if self.dataset_name == "LAW":
            expected_answer = answer[len(input_answer):]
        elif self.dataset_name == "TEST":
            expected_answer = answer[len(input_answer):]
        elif self.dataset_name == "GSM8K":
            idx = answer.find("The final answer is")
            if idx != -1:
                expected_answer = answer[idx:]
                cleaned_answer = answer[:idx]
            else:
                return 0
        elif self.dataset_name == "LEGAL_SCM":
            idx = answer.rfind("最终答案")
            if idx != -1:
                expected_answer = answer[idx:]
                cleaned_answer = answer[:idx]
            else:
                return 0
        elif self.dataset_name == "LEGAL_LQA" or self.dataset_name == "KNOWLEDGE" or self.dataset_name == "CALUATION" or self.dataset_name == "FINEVAL":
            idx = answer.rfind("正确答案")
            if idx != -1:
                expected_answer = answer[idx:]
                cleaned_answer = answer[:idx]
            else:
                return 0
        elif self.dataset_name == "LEGAL_CFM":
            idx = answer.rfind("正确答案")
            if idx != -1:
                expected_answer = answer[idx:]
                cleaned_answer = answer[:idx]
            else:
                return 0
        elif self.dataset_name == "JEJS":
            idx = answer.rfind("最终金额")
            if idx != -1:
                expected_answer = answer[idx:]
                cleaned_answer = answer[:idx]
            else:
                return 0
        elif self.dataset_name == "ZMYC":
            idx = answer.rfind("罪名")
            if idx != -1:
                expected_answer = answer[idx:]
                cleaned_answer = answer[:idx]
            else:
                return 0
        elif self.dataset_name == "AGIEVAL":
            idx = answer.rfind("最终答案")
            if idx != -1:
                expected_answer = answer[idx:]
                cleaned_answer = answer[:idx]
            else:
                return 0
        else:
            raise ValueError("Invalid dataset name")
        for i in [0.2, 0.4, 0.6]:
            input_answer.append(cleaned_answer[:round(i * len(cleaned_answer))])
        
        # input_answer = cleaned_answer[:round(random.uniform(0.2, 0.5) * len(cleaned_answer))]

        num_return = self.mcts_num_last_votes
        if model_type == "domain_model":
            io_input, cleaned_io_output_list = self._context_answer_reward(
                model_type="base_model", question=user_question, context=context, part_answer=input_answer, verbose=verbose
            )
        elif model_type == "base_model":
            io_input, cleaned_io_output_list = self._context_answer_reward(
                model_type="base_model", question=user_question, context=context, part_answer=input_answer, verbose=verbose
            )
        else:
            raise ValueError("model_type must be either 'domain_model' or 'base_model'")
        
        reward_list=[]
        for io_output in cleaned_io_output_list:
            if self.dataset_name == "LAW":
                rouge_score = compute_flzx(io_output, expected_answer)
                reward_list.append(rouge_score)    
            elif self.dataset_name == "GSM8K":
                reward = compute_acc([io_output],[expected_answer])
                reward_list.append(reward)
            elif self.dataset_name == "LEGAL_SCM":
                f1_score = compute_scm(io_output, expected_answer)
                reward_list.append(f1_score)
            elif self.dataset_name == "LEGAL_LQA" or self.dataset_name == "BAOXIAN" or self.dataset_name == "QIHUO" or self.dataset_name == "SHUIWU":
                f1_score = compute_lqa(io_output, expected_answer)
                reward_list.append(f1_score)
            elif self.dataset_name == "LEGAL_CFM":
                f1_score = compute_cfm(io_output, expected_answer)
                reward_list.append(f1_score)
            elif self.dataset_name == "JEJS":
                acc_score = compute_jejs(io_output, expected_answer)
                reward_list.append(acc_score)
            elif self.dataset_name == "TSET":
                rouge_score = compute_flzx(io_output, expected_answer)
                reward_list.append(rouge_score)
            elif self.dataset_name == "ZMYC":
                pred,ref = extract_zmyc(io_output, expected_answer)
                if ref == [['']]:
                    acc_score = 0
                else:
                    acc_score = len(set(ref[0]) & set(pred[0])) / len(set(ref[0]) | set(pred[0]))

                reward_list.append(acc_score)
            elif self.dataset_name == "AGIEVAL":
                acc_score = compute_agi(io_output, expected_answer)
                reward_list.append(acc_score)
            elif self.dataset_name == "KNOWLEDGE" or self.dataset_name == "FINEVAL":
                acc_score = compute_choice(io_output, expected_answer)
                reward_list.append(acc_score)
            elif self.dataset_name == "CALUATION":
                acc_score = compute_choice(io_output, expected_answer)
                reward_list.append(acc_score)
            else:
                raise NotImplementedError(f"dataset {self.dataset_name} is not supported")    
        reward = sum(reward_list) / len(reward_list)
        verbose_print(f"\n\n==========Based on some of the answers, the subsequent answers generated are：{cleaned_io_output_list}，And the node's value is{value}",verbose)
        return reward*value
    
    def generate_rephrased_user_question(self, user_question: str, verbose: bool):
        rephrased_user_question_list=[]
        fewshot_prompt = self.rephrasing_prompt_template
        
        io_input = self.generate_question_parahrased_prompt(user_question, fewshot_prompt,prompt_type=self.base_model_prompt_type)
        verbose_print(f"\n\n==========\n{io_input}",verbose)
        io_output = self.io_base.generate(model_input=io_input, max_tokens=self.max_tokens, num_return=1)[0]
        io_output = io_output.replace('\n', '')
        io_output = io_output.replace('Restatement of the Problem：', '')
        io_output = io_output.replace('Rephrased_Question:', '')
        rephrased_user_question_list.append(io_output)

        return rephrased_user_question_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        user_question: str = None,
        context: str = None,
        answer: str = None,
        max_depth_allowed: int = None,
        # -----------------------------------
        rephrased_user_question: str = None,
        # ---------------------------------------
        potential_answers: List[str] = None,
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()
        # verbose_print(f"Instantiating node with type {node_type}, depth: {depth}, context: {context}, answer: {answer}",verbose)

        #! sanity checks
        assert depth is not None
        assert node_type is not None
        if node_value is not None:
            assert node_value > 0, breakpoint()

        if node_type is Node_Type.USER_QUESTION:
            assert depth == 0
            assert all(
                attr is None
                for attr in [
                    parent,
                    node_value,
                    rephrased_user_question,
                    context,
                    answer,
                ]
            )
            assert all(
                attr is not None
                for attr in [generator, user_question, max_depth_allowed]
            )
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            assert depth >= 1
            assert all(
                attr is None
                for attr in [
                    node_value,
                    generator,
                    user_question,
                    max_depth_allowed,
                ]
            )
            assert all(attr is not None for attr in [parent, rephrased_user_question])
        elif "ANSWER" in str(node_type):
            assert depth > 0
            assert all(
                attr is None
                for attr in [
                    generator,
                    user_question,
                    max_depth_allowed,
                ]
            )
            assert all(attr is not None for attr in [parent, node_value, answer])
        elif node_type is Node_Type.CONTEXT:
            assert depth > 0
            assert all(
                attr is None
                for attr in [
                    generator,
                    user_question,
                    max_depth_allowed,
                ]
            )
            assert all(
                attr is not None for attr in [parent, answer, context]
            )
        elif node_type is Node_Type.REPHRASED_CONTEXT:
            assert depth > 0
            assert all(
                attr is None
                for attr in [
                    generator,
                    max_depth_allowed,
                ]
            )
            assert all(attr is not None for attr in [parent, answer, context, rephrased_user_question])

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.answer = answer

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
        else:
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed

        if node_type is Node_Type.CONTEXT or node_type is Node_Type.REPHRASED_CONTEXT or parent is None:  
            self.context = context
            self.answer = answer
        else:
            self.context = parent.context

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        
        elif "REPHRASED" in str(node_type):
            self.paraphrased = True
            self.user_question = rephrased_user_question
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace = [self.id]
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = parent.solution_trace
            self.solution_trace.append(self.id)

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.BASE_ANSWER: "BA",
            Node_Type.DOMAIN_ANSWER: "DA",
            Node_Type.CONTEXT: "CT",
            Node_Type.REPHRASED_CONTEXT: "RC",
            Node_Type.OST_STEP: "TS",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        def do_action_generate_answers(model_type: str):
            verbose_print(f"----Using {model_type} model to generate answers for node {self.id}... and node type is {self.node_type}", self.verbose)
            (answer_list, value_list) = self.generator.generate_answers(
                model_type=model_type, user_question=self.user_question, paraphrased=self.paraphrased, context=self.context, verbose=self.verbose
            )
            verbose_print(answer_list[0], self.verbose)
            for answer, value in zip(answer_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                if model_type == "base_model":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.BASE_ANSWER,
                            node_value=value,
                            answer=answer,
                        )
                    )
                elif model_type == "domain_model":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DOMAIN_ANSWER,
                            node_value=value,
                            answer=answer,
                        )
                    )
                else:
                    raise ValueError(f"Invalid model type: {model_type}")


        def do_action_generate_context_answers():
            verbose_print(f"---- Generating context for node {self.id}... and node type is {self.node_type}", self.verbose)
            answer, re_context,value = self.generator.generate_context(
                     user_question=self.user_question, context=self.context, answer=self.answer, paraphrased=self.paraphrased, verbose=self.verbose
                )
            
            if self.paraphrased:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_CONTEXT,
                        rephrased_user_question=self.user_question,
                        context=re_context,
                        # context=answer,
                        answer=answer[0],
                        node_value=value[0]
                    ))
            else:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.CONTEXT,
                        context=re_context,
                        answer=answer[0],
                        node_value=value[0]
                    ))

        def do_action_generate_rephrased_user_question():
            verbose_print(f"---- Generating rephrased user question for node {self.id}... and node type is {self.node_type}", self.verbose)

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_question_list = self.generator.generate_rephrased_user_question(
                user_question=self.user_question, verbose=self.verbose
            )
            verbose_print(f"---- The issue of restatement：{rephrased_user_question_list[0]}", self.verbose)
            if self.node_type is Node_Type.CONTEXT:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        context=self.context,
                        node_type=Node_Type.REPHRASED_CONTEXT,
                        rephrased_user_question=rephrased_user_question_list[0],
                        answer=self.answer
                    ))
            else:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=rephrased_user_question_list[0],
                        context=self.context,

                    ))

        if self.node_type is Node_Type.USER_QUESTION:
            # A1
            do_action_generate_answers(model_type="base_model")

            # A2
            do_action_generate_answers(model_type="domain_model")

            # A3
            do_action_generate_context_answers()

            # A5
            do_action_generate_rephrased_user_question()

        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            # A1
            do_action_generate_answers(model_type="base_model")

            # A2
            do_action_generate_answers(model_type="domain_model")

            # A3
            do_action_generate_context_answers()

        elif (self.node_type is Node_Type.BASE_ANSWER) or (self.node_type is Node_Type.DOMAIN_ANSWER):
            raise ValueError("DIRECT_ANSWER node cannot create children!!")

        elif self.node_type is Node_Type.CONTEXT:
            # A1
            do_action_generate_answers(model_type="base_model")
            #A2
            do_action_generate_answers(model_type="domain_model")
            #A3
            do_action_generate_context_answers()
            #A5
            do_action_generate_rephrased_user_question()
        elif self.node_type is Node_Type.REPHRASED_CONTEXT:
            # A1
            do_action_generate_answers(model_type="base_model")
            #A2
            do_action_generate_answers(model_type="domain_model")
            #A3
            do_action_generate_context_answers()
        else:
            raise ValueError(f"Node type {self.node_type} is not supported!!")

        assert self.children
        return self.children

    # Determine whether the current node is a valid leaf node.
    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return  ("CONTEXT" in str(self.node_type) and reach_terminal_context(self.answer)
        ) or "ANSWER" in str(self.node_type)

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_context(self.subquestion, self.user_question)
            )
            # or (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            if self.node_type == Node_Type.BASE_ANSWER or self.node_type == Node_Type.CONTEXT or self.node_type == Node_Type.REPHRASED_CONTEXT:
                reward = self.generator.generate_reward(model_type="base_model", user_question=self.user_question, context=self.context, answer=self.answer, verbose=self.verbose, value = self.node_value,node_type=self.node_type)
            elif self.node_type == Node_Type.DOMAIN_ANSWER:
                reward = self.generator.generate_reward(model_type="domain_model", user_question=self.user_question, context=self.context, answer=self.answer, verbose=self.verbose, value = self.node_value,node_type=self.node_type)
            else:
                raise Exception("Invalid answer node")
            return reward
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASED_USER_QUESTION


def search_for_answers(args, user_question: str, generator: Generator):

    logger = logging.getLogger("mcts")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    verbose_print(
        f"********************* Searching for answers to question ********************* ", args.verbose
    )

    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #create root_node
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        user_question=user_question,
        max_depth_allowed=args.max_depth_allowed,
    )

    process_answer_list = []
    process_context_list=[]

    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):

        selected_node = mcts_searcher.do_rollout(root_node, i)

        mcts_searcher.verbose_draw_tree(root_node, verbose=args.verbose, selected_node=selected_node)

    answer = mcts_searcher.select_final_solution(root_node, verbose=args.verbose)

    return  answer