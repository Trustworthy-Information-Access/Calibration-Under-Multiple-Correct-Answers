import argparse
import collections
import json
import copy
import os
import re
import logging
import string
import regex
import unicodedata
from tqdm import tqdm
from nltk.corpus import stopwords
import base64
import numpy as np
# from PIL import Image
from io import BytesIO

logger = logging.getLogger()

def output_logprobs(completion, stream):
    # Extract logprobs from a stream or completion object
    if stream:
        res = []
        for chunk in completion:
            if chunk.choices:
                if chunk.choices[0].logprobs is not None:
                    print(chunk.choices[0].logprobs.content[0].logprob)
                    res.append(chunk.choices[0].logprobs.content[0].logprob)
        return res
    return [prob.logprob for prob in completion.choices[0].logprobs.content]

def output_completion(completion, stream, useprob):
    # Get content and optionally logprobs from completion or stream
    if stream:
        res = ""
        logprobs = []
        for chunk in completion:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    res += content
                if useprob:
                    if chunk.choices[0].logprobs is not None:
                        logprobs.append(chunk.choices[0].logprobs.content[0].logprob)
        return res, logprobs
    if useprob:
        logprobs = completion.choices[0].logprobs.logprobs
    else:
        logprobs = None
    return completion.choices[0].message.content, logprobs

def encode_image(image_path):
    # Encode image file in base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def encode_image_add_noise(image_path, noise_level):
    # Encode and add Gaussian noise to image (base64 encoded)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(scale=noise_level/100, size=img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        result_img = Image.fromarray(noisy_img)
        buffered = BytesIO()
        result_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
def write_json_from_start(path, start, datas):
    # Write json lines to a file starting from a specific line index
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start_idx = start - 1
    lines = lines[:start_idx]
    for data in datas:
        lines.append(json.dumps(data) + "\n")
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def read_json(path):
    # Read a jsonl file to a list
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    # Write a list of dicts as jsonl to file
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def remove_punc(text):
    # Replace all punctuation with spaces
    exclude = set(string.punctuation)
    return "".join([ch if ch in text and ch not in exclude else ' ' for ch in text])

def is_digital(text):
    # Check if string contains only digits
    return text.isdigit()

def remove_stopwords(text):
    # Remove English stop words from a list of tokens
    words = stopwords.words('english')
    text = [w for w in text if w not in words]
    return text 

def context_len(data):
    # Print average and median length in words for documents in data
    len_list = []
    for sample in data:
        len_list.append(len(remove_punc(sample['dpr_ctx'][0]).split()))
    len_list.sort()
    print(f'average len: {sum(len_list) / len(len_list)}')
    print(f'median len: {len_list[int(len(len_list) / 2)]}')

def get_judge(data, judge_data):
    # Match answers to patterns and assign judge result
    assert len(data) == len(judge_data)
    pattern = ['both', 'none', 'answer 1', 'answer 2', 'option 1', 'option 2']
    for idx in range(len(data)):
        flag = 0
        for p in pattern:
            if has_answer([p], judge_data[idx]['Res']):
                data[idx]['judge'] = p
                flag = 1
                break
        if flag == 0:
            data[idx]['judge'] = 'none'
    return data

def get_clean(data, clean_data):
    # Assign 'clean_pred' field to 'pred' value
    assert len(data) == len(clean_data)
    for idx in range(len(data)):
        data[idx]['clean_pred'] = data[idx]['pred']
    return data

def get_data_before_and_after_prompt(origin_data, prompt_data, criterion):
    # Filter data based on change between origin and prompt according to criterion
    new_res = []
    for sample in origin_data:
        if criterion == 'same':
            if sample['Giveup_origin'] == prompt_data[sample['nq_idx']]['Giveup']:
                new_res.append(sample)
        else:
            if sample['Giveup_origin'] != prompt_data[sample['nq_idx']]['Giveup']:
                new_res.append(sample)
    return new_res

def get_data_before_and_after_evidence(origin_data, prompt_data, criterion):
    # Filter data based on change between origin and evidence according to criterion
    new_res = []
    for idx in range(len(origin_data)):
        sample = origin_data[idx]
        if 'info' in sample:
            continue
        if criterion == 'same':
            if sample['Giveup'] == prompt_data[idx]['Giveup']:
                new_res.append(sample)
        else:
            if sample['Giveup'] != prompt_data[idx]['Giveup']:
                new_res.append(sample)
    print(len(new_res))
    return new_res

def get_data_after_judge(data, judge_data):
    # Assign 'Giveup' from judge_data to data
    print(len(data))
    print(len(judge_data))
    assert len(data) == len(judge_data)
    for idx in range(len(data)):
        if 'info' in data[idx]:
            continue
        data[idx]['Giveup'] = judge_data[idx]['Giveup']
    return data

def judge_again(data):
    # Regenerate 'Giveup' using a stricter rule set
    for idx in range(len(data)):
        data[idx]['Giveup'] = deal_judge_new(data[idx]['Res'])
    return data

def merge_qa_evidence(qa_data, wrong_evidence_data, right_evidence_data):
    """
    Add 'wevidence' and 'revidence' fields to qa_data based on respective evidence
    """
    assert len(qa_data) == len(wrong_evidence_data)
    for idx in range(len(qa_data)):
        if 'info' in qa_data[idx]:
            continue
        qa_data[idx]['wevidence'] = wrong_evidence_data[idx]['Res']
        qa_data[idx]['revidence'] = right_evidence_data[idx]['Res']
    return qa_data
    
def compute_has_answer(ref_data, qa_data):
    # Compute if the answer exists in the QA result
    assert len(ref_data) == len(qa_data)
    for idx in range(len(qa_data)):
        if 'info' in qa_data[idx]:
            continue
        qa_data[idx]['has_answer'] = has_answer(ref_data[idx]['reference'], qa_data[idx]['Res'])
    return qa_data

import unicodedata
import re
import string

def _normalize_answer(s):
    # Standard text normalization (lowercase, remove punc, accents, articles, etc)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def remove_accents(text):
        # Remove accents using Unicode decomposition
        return "".join(ch for ch in unicodedata.normalize("NFD", text)
                       if unicodedata.category(ch) != "Mn")

    return white_space_fix(remove_articles(remove_punc(lower(remove_accents(s)))))

from rapidfuzz import fuzz

def has_answer(answers, text, match_type="string", threshold=90):
    """
    Return 1 if any answer is contained (fuzzy match) in text, otherwise 0.
    """
    text = str(text).strip().lower()
    for single_answer in answers:
        single_answer_str = str(single_answer).strip().lower()
        similarity = fuzz.partial_ratio(single_answer_str, text)
        if similarity >= threshold:
            return 1
    return 0

def EM_compute(answer_list, prediction):
    # Exact match (normalized) between prediction and answer_list
    return max([int(_normalize_answer(prediction) == _normalize_answer(ground_truth)) for ground_truth in answer_list])

def F1_compute(answers, pred):
    # Calculate F1-score between prediction and each reference, return max
    def get_tokens(s):
        if not s: return []
        return _normalize_answer(s).split()

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, F1 is 1 if both empty, else 0
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    return max([compute_f1(x, pred) for x in answers])

def deal_judge(pred):
    # Return True if prediction indicates no answer, otherwise False
    if pred is None:
        return True
    if has_answer([
        "unknown", "no specific answer", "not provide", "cannot answer", 
        "no information provided", "no answer", "not contain", "no definitive answer"], pred):
        return True
    return False

def deal_judge_new(pred):
    # Extended no-answer rules (including apologetic or uncertain phrases)
    if pred is None:
        return True
    if has_answer([
        "sorry", "apologize", "apologies", "uncertain", "false", "no", 'unsure', 
        "cannot", "unknown", "no specific answer", "not provide", "cannot answer", 
        "no information provided", "no answer", "not contain", "no definitive answer"], pred):
        return True
    return False

def deal_judge_not_correct(pred):
    # Return True if prediction contains negative/corrections
    if pred is None:
        return True
    if has_answer([
        "no", "not correct", "incorrect", "not factually correct", "cannot"], pred):
        return True
    return False

def deal_no_info(pred):
    # Return True if prediction contains "no info" keywords
    if pred is None:
        return True
    if has_answer([
        "cannot", "unknown", "provide", 'information', 'assistant', 
        'artificial', 'unsure', 'robot'], pred):
        return True
    return False

def deal_answer(pred, answers):
    # Return EM and F1 scores for prediction and references
    if pred is None:
        return 0, 0
    if pred.lower().startswith("answer:"):
        pred = pred[7:]
    return EM_compute(answers, pred), F1_compute(answers, pred)

def deal_post(pred):
    # Determine final status (giveup or istrue) from prediction
    giveup, istrue = True, None
    if pred is None:
        return giveup, istrue
    if has_answer([
        "uncertain", "unclear", "not clear", "unknown", "partially correct", "partially incorrect", 
        "not correct", "cannot determine", "cannot answer", "not incorrect", "incomplete"], pred):
        giveup = True
    elif has_answer(["correct", "true"], pred):
        giveup, istrue = False, True
    elif has_answer(["incorrect", "false"], pred):
        giveup, istrue = False, False
    else:
        giveup = True
    return giveup, istrue

def str2paras(s):
    # Convert text into a list of non-empty paragraphs with prefix
    if s is None:
        return None
    paras = []
    for text in s.split('\n'):
        if text.strip() != '':
            paras.append(": " + text)
    return paras

def load_source(file):
    # Read jsonl source file to list
    data = []
    f = open(file, 'r', encoding='utf-8')
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data

if __name__ == '__main__':
    print(len(_normalize_answer("2013 ()")))
