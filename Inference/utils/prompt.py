import json

def read_json(path):
    # Read data from a JSONL file line by line
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

# Dictionary containing different prompting templates for tasks
prompt_dict = {
    'eval': {
        'none': 'Question: {question}\nGround truth answer: {answer}\n Statement: {state} \n',
        'ra': 'not avaliable now', 
        'tail': '\nYour Judgement:',
    },
    'qa': {
        'none': 'Answer the following question based on your internal knowledge.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'qa_short': {
        'none' : """
        Return only the answer based on your internal knowledge without any explanation or extra text. 
        Answer in one or a few words. 
        Question: {question}{paras}{prediction}

        """,
        'tail': '\nAnswer: ',
    },
    'qa_short_vanilla_verb': {
        'none': """
        Return only the answer and your confidence (as a decimal number between 0 and 1, 
        rounded to two decimal places). 
        Do not give any explanation or extra text. 
        Answer format: "<answer> | <number>"
        Example: "Paris | 0.92"
        Answer in one or a few words.
        Question: {question}{paras}{prediction}
        """,
        'tail': '\nAnswer: ',
    },
    'qa_short_topk_verb': {
        'none': """
        Return the top 5 possible answers and your confidence for each 
        (as a decimal number between 0 and 1, rounded to two decimal places). 
        Do not give any explanation or extra text. 
        Answer format (each on a new line): "<answer> | <number>"
        Example:
        Paris | 0.92
        Lyon | 0.08
        Question: {question}{paras}{prediction}
        """,
        'tail': '\nAnswer: ',
    },
    'qa_short_output_all': {
        'none': """
        Return only the answer(s) based on your internal knowledge, without any explanation or extra text.
        If there are multiple valid answers, list **all** of them separated by commas.
        Each answer should be concise (a single word or short phrase).
        Question: {question}{paras}{prediction}
        """,
        'tail': '\nAnswer: ',
    },
    'qa_short_vanilla_verb_output_all': {
        'none': """
        Return only the answer(s) and your confidence score (as a decimal number between 0 and 1, 
        rounded to two decimal places), without any explanation or extra text.
        If multiple valid answers exist, list all of them separated by commas in a single line.
        Answer format: "<answer1, answer2, ...> | <number>"
        Example: "Paris, Lyon | 0.92"
        Question: {question}{paras}{prediction}
        """,
        'tail': '\nAnswer: ',
    },
    'qa_short_topk_verb_output_all': {
        'none': """
        Return the top 5 possible answer groups and their shared confidence scores 
        (each as a decimal number between 0 and 1, rounded to two decimal places).
        Each group may contain multiple answers separated by commas.
        Write each answer-confidence pair on a **new line**.
        Do not include any explanation or extra text.
        Answer format (each line): "<ans1, ans2, ...> | <score>"
        Example:
        Paris, Lyon | 0.82
        Marseille, Nice | 0.12
        Toulouse, Bordeaux | 0.06
        Question: {question}{paras}{prediction}
        """,
        'tail': '\nAnswer: ',
    },
}

# Templates for different model prompts
model_template_dict = {
    'qwen-vl-plus-latest':{
        'prefix':'',
        'end':''
    },
    "gpt-4o" :{
        'prefix':'',
        'end':''
    },
    'qwen-omni-turbo':{
        'prefix': '',
        'end': ''
    },
    'llama2-7b-chat':{
        'prefix': '<s>[INST] <<SYS>>\nYou are a helpful assistant<</SYS>>',
        'end': '[/INST]'
    },
    'llama3-8b-instruct':{
        'prefix': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for answering factual questions<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n',
        'end': "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    'qwen2-7b-instruct':{
        'prefix': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
        'end': '<|im_end|>\n<|im_start|>assistant'
    },
    'Qwen2.5-7B-Instruct':{
        'prefix': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
        'end': '<|im_end|>\n<|im_start|>assistant'
    },
    'llama2-13b-chat':{
        'prefix': '<s>[INST] <<SYS>>\nYou are a helpful assistant<</SYS>>\n\n',
        'end': '[/INST]'
    },
}

# Templates for multi-round prompting for specific models
model_template_dict_for_multi_round = {
    'llama3-8b-instruct':{
        'sys_prefix': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for answering factual questions',
        'user_prefix': '<|start_header_id|>user<|end_header_id|>\n\n',
        'assis_prefix': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'end': "<|eot_id|>\n"
    },
    'qwen2-7b-instruct':{
        'sys_prefix': '<|im_start|>system\nYou are a helpful assistant.',
        'user_prefix': '<|im_start|>user\n',
        'assis_prefix': '<|im_start|>assistant\n',
        'end': "<|im_end|>\n"
    },
}

def get_prompt(sample, args):
    # Generate a single-turn prompt string given a sample and arguments
    paras = ""
    ref_key = 'question'
    prompt = prompt_dict[args.type]['none']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prediction = sample['Res'] if 'post' in args.type else ""
    prompt = prompt.format(question=sample[ref_key], paras=paras, prediction=prediction) + tail
    # Model specific template
    if args.model_name in model_template_dict.keys():
        template_prompt = model_template_dict[args.model_name] 
    else:
        template_prompt = {}
        template_prompt['prefix']=''
        template_prompt['end']=''
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt

def get_prompt_for_multi_round(sample, args):
    # Build multi-round prompt conversation for specific models
    """
    Two-round (post):
        - factual question
        - response
        - determine right
    Three-round (post_multi_round):
        - factual question
        - response
        - generate 10 answers
        - 10 answers
        - determine right
    """
    prompt = ''
    template_prompt = model_template_dict_for_multi_round[args.model_name]
    # System role for chat
    prompt += template_prompt['sys_prefix']
    prompt += template_prompt['end']
    if args.type == 'qa_post':
        sample['question'] = sample['question'][:2] # Only two elements needed for qa_post
    for idx in range(len(sample['question'])):
        if idx % 2 == 0:
            # User message
            prompt += template_prompt['user_prefix']
            prompt += sample['question'][idx]
            prompt += template_prompt['end']
        else:
            # Assistant message
            prompt += template_prompt['assis_prefix']
            prompt += sample['question'][idx]
            prompt += template_prompt['end']
    # Add judging instruction
    prompt += template_prompt['user_prefix']
    prompt += f'Please determine whether your response [{sample["question"][1]}] contains the correct answer. If yes, respond with "certain." If it is incorrect, respond with "uncertain." Start your response with "certain" or "uncertain" and do not give any other words.'
    prompt += template_prompt['end']
    prompt += template_prompt['assis_prefix']
    return prompt

def get_evaluate_output_prompt(question, answers, state, args):
    # Generate a prompt string to evaluate model output
    prompt = (
        "{question}"
        "Ground truth answer: {answer}\n"
        "Statement: {statement}\n"
        "Your Judgement:\n"
    )
    prompt = prompt.format(question=question, answer=str(answers), statement=state)
    return prompt

def get_prompt_multiq(sample, args):
    # Generate multiple prompts for multiple questions in a sample
    questions = sample['multiple_questions']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prompts = []
    for question in questions:
        prompt = prompt_dict[args.type]['none']
        prompt = prompt.format(question=question, paras='', prediction='') + tail
        if args.model_name in model_template_dict.keys():
            template_prompt = model_template_dict[args.model_name] 
        else:
            template_prompt = {}
            template_prompt['prefix']=''
            template_prompt['end']=''
        prompt = template_prompt['prefix'] + prompt + template_prompt['end']
        prompts.append(prompt)
    return prompts

def get_prompt_with_disc(sample, description, args):
    # Generate a prompt requiring extra description/context 
    question = sample['question']
    prompt = prompt_dict['vqa_description']['none']
    tail = prompt_dict['vqa_description']['tail']
    prompt = prompt.format(question=question, description=description['Res']) + tail
    if args.model_name in model_template_dict.keys():
        template_prompt = model_template_dict[args.model_name] 
    else:
        template_prompt = {}
        template_prompt['prefix']=''
        template_prompt['end']=''
    prompt = template_prompt['prefix'] + prompt + template_prompt['end']
    return prompt

if __name__ == '__main__':
    # Simple main example: read first entry and run multi-round prompt creation
    model_name = 'qwen7b'
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    mode = 'test'
    dataset = 'nq'
    out_path = f'{base_dir}/{dataset}/multi_round/{dataset}_{mode}_{model_name}.jsonl'
    data = read_json(out_path)
    get_prompt_for_multi_round(data[0], {'model_name': 'llama3-8b-instruct'})

