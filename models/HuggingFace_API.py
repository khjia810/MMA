import torch
from transformers import (
    GenerationConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def load_model(ckpt, use_causal=True) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if use_causal:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModel.from_pretrained(
            ckpt,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    return tokenizer, model

def generate_with_huggingface_model(
    model, tokenizer, input=None, temperature=0.2, top_p=0.7, top_k=15, num_beams=1, max_new_tokens=2048, **kwargs
):
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_length = input_ids.size()[1]
    outputs = []
    for seq in generation_output.sequences:
        generated_tokens = seq[input_length:]
        output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        outputs.append(output)
    return outputs

if __name__ == "__main__":
    model_path = "/data/pretrained_models/wisdom"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    
    response = generate_with_huggingface_model(
        model, 
        tokenizer, 
        input="1+1=?"
    )
    print(response)
