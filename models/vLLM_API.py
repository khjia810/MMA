import torch
import multiprocessing as mp
from multiprocessing import Manager
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def model_worker(request_queue, response_queue, model_ckpt, device, seed, dtype):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    # os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    if "LLM" in model_ckpt:
        llm = LLM(
            model=model_ckpt,
            max_model_len=4096,
            dtype=dtype,
            tensor_parallel_size=1,
            seed=seed,
            trust_remote_code=True,
            # max_num_seqs=256,
            swap_space=16,
        )
    elif "law" or "wisdom" in model_ckpt:
        llm = LLM(
            model=str(model_ckpt),
            max_model_len=4096,
            dtype=dtype,
            tensor_parallel_size=1,
            seed=seed,
            trust_remote_code=True,
            # max_num_seqs=256,
            swap_space=16,
        )
    elif "XuanYuan" or "Phi" in model_ckpt:
        llm = LLM(
            model=model_ckpt,
            max_model_len=8192,
            dtype="float16",
            tensor_parallel_size=1,
            seed=seed,
            trust_remote_code=True,
            # max_num_seqs=256,
            swap_space=16,
        )
    else:
        llm = LLM(
            model=model_ckpt,
            max_model_len=16384,
            dtype=dtype,
            tensor_parallel_size=1,
            seed=seed,
            trust_remote_code=True,
            # max_num_seqs=256,
            swap_space=16,
        )
    while True:
        request = request_queue.get()
        if request == "STOP":
            break

        prompts, sampling_params_dict = request
        sampling_params = SamplingParams(**sampling_params_dict)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        response = []
        for o in outputs:
            candidate_texts = [candidate.text for candidate in o.outputs] if o.outputs else [""]
            response.append({
                "outputs": [{"text": text} for text in candidate_texts]
            })
        
        response_queue.put(response)
    
    del llm
    torch.cuda.empty_cache()

class ModelProxy:
    def __init__(self, model_ckpt, seed, dtype, device):
        self.manager = Manager()
        self.request_queue = self.manager.Queue()
        self.response_queue = self.manager.Queue()
        self.process = mp.Process(
            target=model_worker,
            args=(self.request_queue, self.response_queue, 
                  model_ckpt, device, seed, dtype),
        )
        self.process.start()

    def generate(self, input, sampling_params, **kwargs):
        if isinstance(input, str):
            input = [input]

        params_dict = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "repetition_penalty": sampling_params.repetition_penalty,
            "max_tokens": sampling_params.max_tokens,
            "n": sampling_params.n,
            "logprobs": sampling_params.logprobs,
            "stop": sampling_params.stop,
        }
        
        self.request_queue.put((input, params_dict))
        raw_response = self.response_queue.get()

        results = []
        for item in raw_response:
            if "outputs" in item and len(item["outputs"]) > 0:

                for output in item["outputs"]:
                    results.append(output["text"])
            else:
                results.append("")
        return results

    def shutdown(self):
        self.request_queue.put("STOP")
        self.process.join()

def generate_with_vllm_model(model, input, sampling_params):
    return model.generate(input, sampling_params)