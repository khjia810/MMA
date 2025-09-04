from jinja2 import Template
import functools

def template(prompt_type:str):
    if prompt_type == "llama3.1":
        user_prompt_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        assistant_prompt_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>"
        generation_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif prompt_type == "baichuan":
        user_prompt_template = "<reserved_195>{}"
        assistant_prompt_template = "<reserved_196>{}"
        generation_prefix = "<reserved_196>"
    elif prompt_type == "xuanyuan":
        user_prompt_template = "Human: {}"
        assistant_prompt_template = " Assistant:{}"
        generation_prefix = " Assistant:"
    elif prompt_type == "qwen":
        user_prompt_template = "<|im_start|>user\n{}<|im_end|>"
        assistant_prompt_template = "<|im_start|>assistant\n{}<|im_end|>"
        generation_prefix = "<|im_start|>assistant\n"
    elif prompt_type == "mistral":
        user_prompt_template = "<s>[INST]{}"
        assistant_prompt_template = "[/INST]{}</s>"
        generation_prefix = "[INST]"
    elif prompt_type == "empty":
        user_prompt_template = "{}"
        assistant_prompt_template = "{}"
        generation_prefix = ""
    elif prompt_type == "lawyer":
        user_prompt_template = "<s>{}</s>"
        assistant_prompt_template = "<s>{}</s>"
        generation_prefix = "<s>"
    elif prompt_type == "wisdom":
        user_prompt_template = "</s>Human:{}"
        assistant_prompt_template = "</s>Assistant:{}"
        generation_prefix = "</s>Assistant:" 
    elif prompt_type == "Phi":
        user_prompt_template = "<|user|>\n{}<|end|>\n"
        assistant_prompt_template = "<|assistant|>\n{}<|end|>\n"
        generation_prefix = "<|assistant|>\n" 
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return user_prompt_template, assistant_prompt_template, generation_prefix
def fewshot_prompt(question, few_shot_prompt, paraphrased, prompt_type="llama3.1"):

    # 配置对话模板
    user_prompt_template, assistant_prompt_template, generation_prefix = template(prompt_type)

    # 定义核心模板
    if paraphrased:
        instruction_template = (
            "请根据文书中的条件仔细计算涉及的犯罪总金额。你需要根据我的格式要求先给出解析，然后再以“最终金额:”为开头给出最终金额。\n"
            "文书: {{ 问题 }}\n"
            "你的答案格式应该为: “回答:[解析]。最终金额:[最终金额]”。其中[解析]是金额的筛选与计算过程，[最终金额]是文书中涉及的犯罪总金额。\n"
        )
    else:
        instruction_template = (
            "请仔细计算文书中涉及的犯罪总金额。你需要根据我的格式要求先给出解析，然后再以“最终金额:”为开头给出最终金额。\n"
            "文书: {{ 问题 }}\n"
            "你的答案格式应该为: “回答:[解析]。最终金额:[最终金额]”。其中[解析]是金额的筛选与计算过程，[最终金额]是文书中涉及的犯罪总金额。\n"
        )
    
    answer_template = "回答: {{ 回答 }}"

    input_final_prompts = ""
    
    # 渲染few-shot示例
    for few_shot in few_shot_prompt:
        
        # 渲染用户提示
        user_content = Template(instruction_template).render(
            问题=few_shot["问题"]
        )
        input_final_prompts += user_prompt_template.format(user_content)
        
        # 渲染助手回复
        assistant_content = Template(answer_template).render(
            回答=few_shot["回答"]
        )
        input_final_prompts += assistant_prompt_template.format(assistant_content)

    # 添加当前问题
    current_user_content = Template(instruction_template).render(
        问题=question
    )
    input_final_prompts += user_prompt_template.format(current_user_content)
    
    # 添加生成前缀
    input_final_prompts += generation_prefix

    return input_final_prompts

def fewshot_context_prompt(question, context, few_shot_prompt, paraphrased, prompt_type="llama3.1"):

    # 配置对话模板
    user_prompt_template, assistant_prompt_template, generation_prefix = template(prompt_type)

    # 定义核心模板
    if paraphrased:
        instruction_template = (
            "请结合已有分析，根据文书中的条件仔细计算涉及的犯罪总金额。你需要根据我的格式要求先给出解析，然后再以“最终金额:”为开头给出最终金额。\n"
            "文书: {{ 问题 }}\n"
            "已有分析: {{ 已有分析 }}\n"
            "你的答案格式应该为: “回答:[解析]。最终金额:[最终金额]”。其中[解析]是金额的筛选与计算过程，[最终金额]是文书中涉及的犯罪总金额。\n"
        )
    else:
        instruction_template = (
            "请结合已有分析仔细计算文书中涉及的犯罪总金额。你需要根据我的格式要求先给出解析，然后再以“最终金额:”为开头给出最终金额。\n"
            "文书: {{ 问题 }}\n"
            "已有分析: {{ 已有分析 }}\n"
            "你的答案格式应该为: “回答:[解析]。最终金额:[最终金额]”。其中[解析]是金额的筛选与计算过程，[最终金额]是文书中涉及的犯罪总金额。\n"
        )
    
    answer_template = "回答: {{ 回答 }}"

    input_final_prompts = ""
    
    # 渲染few-shot示例
    for few_shot in few_shot_prompt:
        
        # 渲染用户提示
        user_content = Template(instruction_template).render(
            问题=few_shot["问题"],
            已有分析=few_shot["已有分析"]
        )
        input_final_prompts += user_prompt_template.format(user_content)

        
        # 渲染助手回复
        assistant_content = Template(answer_template).render(
            回答=few_shot["回答"]
        )
        input_final_prompts += assistant_prompt_template.format(assistant_content)

    # 添加当前问题
    current_user_content = Template(instruction_template).render(
        问题=question,
        已有分析=context
    )
    input_final_prompts += user_prompt_template.format(current_user_content)
    
    # 添加生成前缀
    input_final_prompts += generation_prefix

    return input_final_prompts
def context_prompt(question, context_cot_prompt, paraphrased, prompt_type="llama3.1"):

    # 配置对话模板
    user_prompt_template, assistant_prompt_template, generation_prefix = template(prompt_type)
    
    if paraphrased:
        instruction_template = (
            "请分析下面文书中的条件，分析该文书中每次的简略犯罪事实以及涉及的金额，不需要给出总的犯罪金额。\n"
            "文书: {{ 问题 }}\n"
            "你的答案格式应该为: “分析:[要素]”。其中[要素]是你对该文书的分析内容。\n"
        )
    else:
        instruction_template = (
            "请分析下面的文书，分析该文书中每次的简略犯罪事实以及涉及的金额，不需要给出总的犯罪金额。\n"
            "文书: {{ 问题 }}\n"
            "你的答案格式应该为: “分析:[要素]”。其中[要素]是你对该文书的分析内容。\n"
        )
    
    answer_template = "分析: {{ 分析 }}"

    input_final_prompts = ""
    
    # 渲染few-shot示例
    for few_shot in context_cot_prompt:
        
        # 渲染用户提示
        user_content = Template(instruction_template).render(
            问题=few_shot["问题"]
        )
        input_final_prompts += user_prompt_template.format(user_content)
        
        # 渲染助手回复
        assistant_content = Template(answer_template).render(
            分析=few_shot["分析"]
        )
        input_final_prompts += assistant_prompt_template.format(assistant_content)

    # 添加当前问题
    current_user_content = Template(instruction_template).render(
        问题=question
    )
    input_final_prompts += user_prompt_template.format(current_user_content)
    
    # 添加生成前缀
    input_final_prompts += generation_prefix

    return input_final_prompts

# def context_answer_prompt(question, context_cot_prompt, context, answer, prompt_type="llama3.1"):

#     # 配置对话模板
#     user_prompt_template, assistant_prompt_template, generation_prefix = template(prompt_type)

#     # 定义核心模板
#     instruction_template = (
#         "下面的待审回答是结合原始分析对文书中的犯罪金额进行的计算。请根据待审回答审查原始分析的合理性和正确性，然后按审查要求以及输出格式生成审查结果。\n"
#         "文书: {{ 问题 }}\n"
#         "原始分析: {{ 原始分析 }}\n"
#         "待审回答: {{ 待审回答 }}\n"
#         "审查要求: 1. 若待审回答或原始分析中存在漏洞或错误，使用[更正]开头，并给出完整、正确的分析，但不用说明错误原因。 2. 如果完全正确，请使用[无需更改]开头。\n"
#         "输出格式: “[审查结论]。分析:[要素]。”。其中[审查结论]为[更正]、[无需更改]中的一个，[要素]为完整的问题分析（该文书中每次的简略犯罪事实以及涉及的金额，且分析中不应给出总的犯罪金额。），且不需要给出对待审答案的分析。\n"
#     )
    
#     answer_template = "审查结果: {{ 审查结果 }}"

#     input_final_prompts = ""
    
#     # 渲染few-shot示例
#     for few_shot in context_cot_prompt:
        
#         # 渲染用户提示
#         user_content = Template(instruction_template).render(
#             问题=few_shot["问题"],
#             原始分析=few_shot["原始分析"], 
#             待审回答=few_shot["待审回答"], 
#             审查结果=few_shot["审查结果"]
#         )
#         input_final_prompts += user_prompt_template.format(user_content)

#         assistant_content = Template(answer_template).render(
#             审查结果=few_shot["审查结果"]
#         )
#         input_final_prompts += assistant_prompt_template.format(assistant_content)

#     # 添加当前问题
#     current_user_content = Template(instruction_template).render(
#         问题=question,
#         原始分析=context, 
#         待审回答=answer, 
#     )
#     input_final_prompts += user_prompt_template.format(current_user_content)
    
#     # 添加生成前缀
#     input_final_prompts += generation_prefix

#     return input_final_prompts


def context_answer_prompt(question, context_cot_prompt, context, answer, prompt_type="llama3.1"):

    # 配置对话模板
    user_prompt_template, assistant_prompt_template, generation_prefix = template(prompt_type)

    # 定义核心模板
    instruction_template = (
        "下面的待审回答是结合其他模型对该问题的回答对文书中的犯罪金额进行的计算。请根据待审回答审查原始分析的合理性和正确性，然后按审查要求以及输出格式生成审查结果。\n"
        "文书: {{ 问题 }}\n"
        "原始分析: {{ 原始分析 }}\n"
        "待审回答: {{ 待审回答 }}\n"
        "审查要求: 1. 若待审回答或原始分析中存在漏洞或错误，使用[更正]开头，并给出完整、正确的分析，但不用说明错误原因。 2. 如果完全正确，请使用[无需更改]开头。\n"
        "输出格式: “[审查结论]。分析:[要素]。”。其中[审查结论]为[更正]、[无需更改]中的一个。\n"
    )
    
    answer_template = "审查结果: {{ 审查结果 }}"

    input_final_prompts = ""
    
    # 渲染few-shot示例
    for few_shot in context_cot_prompt:
        
        # 渲染用户提示
        user_content = Template(instruction_template).render(
            问题=few_shot["问题"],
            原始分析=few_shot["原始分析"], 
            待审回答=few_shot["待审回答"], 
            审查结果=few_shot["审查结果"]
        )
        input_final_prompts += user_prompt_template.format(user_content)

        assistant_content = Template(answer_template).render(
            审查结果=few_shot["审查结果"]
        )
        input_final_prompts += assistant_prompt_template.format(assistant_content)

    # 添加当前问题
    current_user_content = Template(instruction_template).render(
        问题=question,
        原始分析=context, 
        待审回答=answer, 
    )
    input_final_prompts += user_prompt_template.format(current_user_content)
    
    # 添加生成前缀
    input_final_prompts += generation_prefix

    return input_final_prompts




def question_parahrased_prompt(input_question, few_shot_prompt, prompt_type="llama3.1"):

    # 配置对话模板
    user_prompt_template, assistant_prompt_template, generation_prefix = template(prompt_type)

    instruction_template = (
        "请将文书上下文拆分成条件，帮助我重新表述文书。在重新措辞的文书中，请记住要完整表达原文书中的信息。\n"
        "原文书: {{ 原问题 }}\n"
        )

    answer_template = "问题重述: {{ 问题重述 }}"

    input_final_prompts = ""
    
    # 渲染few-shot示例
    for few_shot in few_shot_prompt:
        
        # 渲染用户提示
        user_content = Template(instruction_template).render(
            原问题=few_shot["原问题"]
        )
        input_final_prompts += user_prompt_template.format(user_content)
        
        # 渲染助手回复
        assistant_content = Template(answer_template).render(
            问题重述=few_shot["问题重述"]
        )
        input_final_prompts += assistant_prompt_template.format(assistant_content)

    # 添加当前问题
    current_user_content = Template(instruction_template).render(
        原问题=input_question
    )
    input_final_prompts += user_prompt_template.format(current_user_content)
    
    # 添加生成前缀
    input_final_prompts += generation_prefix

    return input_final_prompts