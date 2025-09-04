# Licensed under the MIT license.

import sys
import os
import importlib.util
sys.path.append(".")
from models.vLLM_API import generate_with_vllm_model
from models.HuggingFace_API import generate_with_huggingface_model
from typing import List, Dict
from vllm import LLM, SamplingParams


class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tokenizer = tokenizer
        self.model = model

        self.call_counter = 0

    def generate(self, model_input, max_tokens: int, num_return: int):
        if self.api == "vllm":
            response = generate_with_vllm_model(
                self.model,
                input=model_input,
                sampling_params=SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                n=num_return,
                max_tokens=max_tokens,
            ))
            io_output_list = [o for o in response]
            
        elif self.api == "huggingface":
            if isinstance(model_input, str):
                response = generate_with_huggingface_model(
                    self.model,
                    self.tokenizer,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    num_beams=num_return,
                    max_new_tokens=max_tokens,
                )
                io_output_list = [o for o in response]
                self.call_counter += 1
    
            elif isinstance(model_input, list):
                response = generate_with_huggingface_model(
                    self.model,
                    self.tokenizer,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    num_beams=num_return,
                    max_new_tokens=max_tokens,
                )
                io_output_list = [
                    [o.text for o in resp_to_single_input.outputs] for resp_to_single_input in response
                ]
                self.call_counter += 1
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported, it must be one of {self.supported_model_types}")
            
        return io_output_list
