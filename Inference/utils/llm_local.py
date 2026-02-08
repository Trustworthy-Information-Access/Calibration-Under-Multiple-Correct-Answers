from cgitb import reset
import time
import os
import logging
import json
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from utils.prompt import get_evaluate_output_prompt
import torch
import numpy as np
from utils.utils import read_json, write_jsonl
import os
# print("🚨 Is CUDA initialized:", torch.cuda.is_initialized())
from tqdm import tqdm
import re
import asyncio
import aiohttp
import random
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm_asyncio
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed import destroy_model_parallel
except ImportError:
    print("Warning: vLLM not installed. Please install using 'pip install vllm'.")
    LLM = None
    SamplingParams = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

all_choices = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
false_terms = ['false', 'wrong', 'no answer', 'not in']
true_terms = ['true', 'correct', 'has answer']

def format_messages_to_prompt(messages: List[Dict[str, str]], model_name: str = "", max_model_len: int = 4096) -> str:
    """
    Convert message format to a prompt format suitable for vLLM, supporting common chat template formats.
    Specific format correction for llama3-8b-instruct.
    Truncate prompt to max_model_len characters.
    """
    if not messages:
        return ""
    model_name = model_name.lower() if model_name else ""
    # ChatML format
    if any(name in model_name for name in ['qwen']):
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        formatted = formatted[:max_model_len]
        return formatted
    # llama3-8b-instruct format
    elif "llama-3" in model_name or "llama3" in model_name:
        prompt = ""
        # system prompt
        system_msg = None
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
                break
        if system_msg is None:
            system_msg = "You are a helpful assistant."
        prompt += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        prompt += f"{system_msg}<|eot_id|>\n"
        # user/assistant turns
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue
            elif role == "user":
                prompt += "<|start_header_id|>user<|end_header_id|>\n"
                prompt += f"{content}<|eot_id|>\n"
            elif role == "assistant":
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
                prompt += f"{content}<|eot_id|>\n"
        # Append assistant header, waiting for generation
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        prompt = prompt[:max_model_len]
        return prompt
    # Llama2/llama1 format
    elif any(name in model_name for name in ['llama']):
        # Compatible with llama2/llama1 [INST] format
        if len(messages) == 1:
            return messages[0]['content'][:max_model_len]
        formatted = ""
        for i, msg in enumerate(messages):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                formatted += f"System: {content}\n" # Note: Original code logic might be mixed here for llama, keeping strictly as provided but translated.
                # Actually, strictly following the provided snippet logic:
                formatted += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == 'user':
                formatted += f"[INST] {content} [/INST] "
            elif role == 'assistant':
                formatted += f"{content} "
        formatted = formatted[:max_model_len]
        return formatted
    # Default format
    else:
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                formatted += f"System: {content}\n"
            elif role == 'user':
                formatted += f"Human: {content}\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n"
        if not formatted.endswith("Assistant: "):
            formatted += "Assistant: "
        formatted = formatted[:max_model_len]
        return formatted

class LocalGenerator:
    def __init__(self, args):
        self.args = args
        self.model_name = getattr(args, "model_name", None)
        self.model_path = getattr(args, "model_path", None)
        if self.model_name is None:
            raise ValueError("Must provide MODEL_NAME (args.model_name)")
        if self.model_path is None:
            raise ValueError("Must provide MODEL_PATH (args.model_path)")
        self.batch_size = args.batch_size
        self.writed_line = self.args.start_line
        self.ok_line = self.args.start_line
        self.cnt = 0
        self.problist = []
        self.tensor_parallel_size = getattr(args, 'tensor_parallel_size', 8)
        self.max_model_len = getattr(args, 'max_model_len', 4096)
        self.trust_remote_code = getattr(args, 'trust_remote_code', True)
        self.eval_model_name = args.eval_model_name
        self.eval_model_path = args.eval_model_path
        self.gpu_memory_utilization = getattr(args, 'gpu_memory_utilization', 0.9)

        if not self.args.using_host and not self.args.using_api:
            if not self.args.eval_only:
                self.llm = self._initialize_vllm_model()
            else:
                self.eval_llm = self._initialize_vllm_model_eval() 

    def _initialize_vllm_model(self):
        if LLM is None:
            raise ImportError("vLLM not installed. Please install using 'pip install vllm'.")
        try:
            print("🔥 Is CUDA initialized before creating vLLM:", torch.cuda.is_initialized())

            logger.info(f"Initializing vLLM model: {self.model_name} Path: {self.model_path}")
            llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                disable_log_stats=True,
            )
            logger.info("vLLM model initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"vLLM model initialization failed: {e}")
            raise e

    def _initialize_vllm_model_eval(self):
        if LLM is None:
            raise ImportError("vLLM not installed. Please install using 'pip install vllm'.")
        try:
            print("🔥 Is CUDA initialized before creating vLLM:", torch.cuda.is_initialized())

            logger.info(f"Initializing vLLM model: {self.eval_model_name} Path: {self.eval_model_path}")
            llm = LLM(
                model=self.eval_model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                disable_log_stats=True,
            )
            logger.info("vLLM model initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"vLLM model initialization failed: {e}")
            raise e

    def _create_sampling_params(self, temperature=0.0, top_p=1.0, max_tokens=2048):
        if SamplingParams is None:
            raise ImportError("vLLM not installed. Please install using 'pip install vllm'.")
        # llama3-8b-instruct recommends stop token <|eot_id|>
        stop_tokens = None
        if self.model_name and ("llama-3" in self.model_name.lower() or "llama3" in self.model_name.lower()):
            stop_tokens = ["<|eot_id|>"]

        if self.model_name and ("qwen" in self.model_name.lower()):
            # End of turn marker for Qwen ChatML
            stops = ["<|im_end|>"]
            stop_tokens = (stop_tokens or []) + stops
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
        )


    def local_chat(self, messages: List[Dict[str, str]], temperature=0.0, top_p=1.0, max_tokens=2048) -> Optional[str]:
        try:
            prompt = format_messages_to_prompt(messages, self.model_name)
            sampling_params = self._create_sampling_params(temperature, top_p, max_tokens)
            outputs = self.llm.generate([prompt], sampling_params)
            if outputs and len(outputs) > 0:
                # llama3-8b-instruct generated content might end with <|eot_id|>, remove it
                text = outputs[0].outputs[0].text
                if text.endswith("<|eot_id|>"):
                    text = text[:-10]
                return text.strip()
            else:
                return None
        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            return None

    def local_chat_batch(self, messages_list: List[List[Dict[str, str]]], temperature=0.0, top_p=1.0, max_tokens=2048, method="") -> List[Optional[str]]:
        try:
            prompts = [format_messages_to_prompt(messages, self.model_name, getattr(self, "max_model_len", 4096)) for messages in messages_list]
            sampling_params = self._create_sampling_params(temperature, top_p, max_tokens)
            outputs = self.llm.generate(prompts, sampling_params)
            results = []
            for output in outputs:
                if output.outputs and len(output.outputs) > 0:
                    text = output.outputs[0].text
                    if text.endswith("<|eot_id|>"):
                        text = text[:-10]
                    results.append(text.strip())
                else:
                    results.append(None)
            return results
        except Exception as e:
            logger.error(f"Batch local inference failed: {e}")
            return [None] * len(messages_list)

    def local_chat_batch_host(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        method: str = ""
    ) -> List[Optional[str]]:
        """
        Local vLLM host batch interface (port 8000) supporting batch requests and progress bars.
        """
        import requests
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, *args, **kwargs): return x  # fallback if tqdm is unavailable

        api_url = "http://localhost:8000/v1/completions"
        headers = {"Content-Type": "application/json"}
        results = []

        prompts = [format_messages_to_prompt(messages, self.model_name, getattr(self, "max_model_len", 4096)) for messages in messages_list]
        batch_size = 512  # Adjustable. Ensure it doesn't exceed single payload size limit
        total = len(prompts)

        for start in tqdm(range(0, total, batch_size), desc="vLLM API batch infer (host)", ncols=80):
            end = min(start + batch_size, total)
            batch_prompts = prompts[start:end]

            payload = {
                "model": self.model_path,
                "prompt": batch_prompts,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }

            try:
                # Support batch request
                response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()

                if "choices" in data and isinstance(data["choices"], list):
                    # choices length matches batch
                    for choice in data["choices"]:
                        text = choice.get("text", "")
                        if isinstance(text, str):
                            text = text.strip()
                            if text.endswith("<|eot_id|>"):
                                text = text[:-10].strip()
                            results.append(text)
                        else:
                            results.append(None)
                else:
                    # If request succeeds but structure mismatches, fill with None based on batch
                    results.extend([None] * len(batch_prompts))
            except Exception as e:
                logger.error(f"vLLM local API call failed: {e}")
                results.extend([None] * len(batch_prompts))

        return results
  
    def local_chat_batch_eval(self, messages_list: List[List[Dict[str, str]]], temperature=0.0, top_p=1.0, max_tokens=2048) -> List[Optional[str]]:
        try:
            prompts = [format_messages_to_prompt(messages, self.eval_model_name, getattr(self, "max_model_len", 4096)) for messages in messages_list]
            sampling_params = self._create_sampling_params(temperature, top_p, max_tokens)
            outputs = self.eval_llm.generate(prompts, sampling_params)
            results = []
            for output in outputs:
                if output.outputs and len(output.outputs) > 0:
                    text = output.outputs[0].text
                    if text.endswith("<|eot_id|>"):
                        text = text[:-10]
                    results.append(text.strip())
                else:
                    results.append(None)
            return results
        except Exception as e:
            logger.error(f"Batch local inference failed: {e}")
            return [None] * len(messages_list)

    
    def local_chat_batch_eval_host(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048
    ) -> List[Optional[str]]:
        """
        Call local vLLM port (http://localhost:8001/v1/completions)
        Perform batch evaluation inference, changed to batch input, and show progress bar (progress bar displayed during request).
        """
        import requests
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, *args, **kwargs): return x  # Downgrade to normal list if tqdm is not installed

        api_url = "http://localhost:8001/v1/completions"
        headers = {"Content-Type": "application/json"}
        results = []

        try:
            prompts = [format_messages_to_prompt(messages, self.eval_model_name, getattr(self, "max_model_len", 4096)) for messages in messages_list]
            batch_size = 512  # Support larger batch, avoid single request being too large, adjust according to actual needs
            total = len(prompts)

            # Progress bar displays during "request"
            pbar = tqdm(range(0, total, batch_size), desc="vLLM batch inferencing", ncols=80)
            for start in pbar:
                batch_prompts = prompts[start:start + batch_size]
                batch_payload = {
                    "model": self.eval_model_path,
                    "prompt": batch_prompts,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "stream": False,
                }

                try:
                    response = requests.post(api_url, headers=headers, json=batch_payload, timeout=180)
                except Exception as req_err:
                    logger.error(f"❌ Request {api_url} failed: {req_err}")
                    results.extend([None] * len(batch_prompts))
                    continue

                if response.status_code != 200:
                    logger.error(f"❌ Response error ({response.status_code}): {response.text[:200]}")
                    results.extend([None] * len(batch_prompts))
                    continue

                data = response.json()
                choices = data.get("choices", [])
                # Process choices one by one to fill results
                for idx in range(len(batch_prompts)):
                    try:
                        text = choices[idx]["text"] if idx < len(choices) else None
                        if text is not None and text.endswith("<|eot_id|>"):
                            text = text[:-10]
                        results.append(text.strip() if text is not None else None)
                    except Exception as parse_err:
                        logger.error(f"❌ Failed to parse response: {parse_err}")
                        results.append(None)

            # Fill with None if insufficient, truncate if excess
            if len(results) < total:
                results.extend([None] * (total - len(results)))
            elif len(results) > total:
                results = results[:total]
            return results

        except Exception as e:
            logger.error(f"Batch local inference failed: {e}")
            return [None] * len(messages_list)
    
    async def local_chat_batch_eval_api_async(
        self,
        messages_list: List[List[Dict[str, str]]],
        api_key: str,
        api_url: str,
        model: str = "deepseek-r1",
        concurrency: int = 100,
        max_retries: int = 3,
    ) -> List[Optional[str]]:
        """
        🚀 Async Concurrent + Retry version API Call
        ---------------------------------------
        Input: messages_list = List[List[Dict[str,str]]]
        Output: List[Optional[str]]
        """
        semaphore = asyncio.Semaphore(concurrency)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SemanticEvalBot/1.0",
        }

        async def call_api(session, messages):
            """Single request logic"""
            for attempt in range(max_retries):
                try:
                    payload = {"model": model, "messages": messages}
                    async with semaphore:
                        async with session.post(
                            api_url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=90),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                return data["choices"][0]["message"]["content"]
                            else:
                                if attempt < max_retries - 1:
                                    await asyncio.sleep((2**attempt) + random.random())
                                else:
                                    return None
                except Exception:
                    if attempt < max_retries - 1:
                        await asyncio.sleep((2**attempt) + random.random())
                    else:
                        return None
            return None

        async with aiohttp.ClientSession() as session:
            tasks = [call_api(session, msg) for msg in messages_list]
            # print(f"⚙️ Starting remote API inference: Total {len(tasks)} tasks, concurrency={concurrency}")
            results = await tqdm_asyncio.gather(*tasks, desc="Remote inferencing")
            return results

    def local_chat_batch_eval_api(
        self,
        messages_list: List[List[Dict[str, str]]],
        api_key: str = "sk-tTo3MNJgAsRIvFgyuRCWfUKSVkBpIgBtPZi7yKTGGAmspl5D",
        api_url: str = "http://123.129.219.111:3000/v1/chat/completions",
        model: str = "gpt-4o-mini",
        concurrency: int = 500,
    ) -> List[Optional[str]]:
        """Synchronous entry (wrapping async call)"""
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(
            self.local_chat_batch_eval_api_async(
                messages_list, api_key, api_url, model, concurrency
            )
        )

    import re

    def extract_res_and_confidence(self, res_text):
        """
        Extract only answer content and confidence from format "Content | 0.9". Return: (res, confidence)
        """
        if not isinstance(res_text, str):
            return res_text, None

        # Preprocessing: Remove escaped quotes and surrounding spaces
        res_text = res_text.strip().strip('"').strip("'").replace('\\"', '').replace("\\'", "")

        # Regex match "Content | 0.9"
        match = re.match(r'^(.*)\|\s*([0-1](?:\.\d+)?)\s*$', res_text)
        if match:
            return match.group(1).strip(), float(match.group(2))
        return res_text.strip(), None

    def extract_topk_res_and_confidence(self, raw_output):
        """
        Process topk_verb output, return answer with max confidence, confidence, and raw output.
        Input format is multi-line, each line like "Content | 0.9"
        Supports raw_output as str or list
        Return: (res, confidence, raw)
        """
        if isinstance(raw_output, list):
            lines = [line.strip() for line in raw_output if isinstance(line, str) and line.strip()]
        elif isinstance(raw_output, str):
            lines = [line.strip() for line in raw_output.strip().split('\n') if line.strip()]
        else:
            return raw_output, None, raw_output

        best_res = None
        best_conf = -1
        for line in lines:
            res, conf = self.extract_res_and_confidence(line)
            if conf is not None and conf > best_conf:
                best_conf = conf
                best_res = res
        # If no confidence found, take the first line
        if best_res is None and lines:
            best_res, best_conf = self.extract_res_and_confidence(lines[0])
        return best_res, best_conf, raw_output

    def get_res_file(self):
        all_data = self.data.data
        res = []
        begin = 0
        logger.info(f'All data length: {len(all_data)}')
        qa_prompts = []
        qa_questions = []
        qa_references = []
        qa_results = []
        for idx in range(len(all_data)):
            if idx not in self.data.idxs:
                res.append(all_data[idx])
            else:
                if begin >= len(self.outputs):
                    break
                if 'qa' in self.args.type:
                    qa_prompts.append(self.data[begin])
                    qa_questions.append(all_data[idx]['question'])
                    qa_references.append(all_data[idx]['reference'])
                    qa_results.append(self.outputs[begin]['Res'])
                begin += 1

        qa_ptr = 0
        for idx in range(len(all_data)):
            if idx not in self.data.idxs:
                continue
            res_sample = {}
            if 'qa' in self.args.type:
                res_sample['qa_prompt'] = qa_prompts[qa_ptr]
                res_sample['question'] = qa_questions[qa_ptr]
                res_sample['reference'] = qa_references[qa_ptr]
                if getattr(self.args, "using_vanilla_verb", False):
                    # Extract answer and confidence
                    if self.args.using_api:
                        raw_output = qa_results[qa_ptr]
                    else:
                        raw_output = qa_results[qa_ptr][0]
                    # Compatible with case where raw_output is list
                    if isinstance(raw_output, list):
                        res_list = []
                        conf_list = []
                        for line in raw_output:
                            res_text, conf = self.extract_res_and_confidence(line)
                            res_list.append(res_text)
                            conf_list.append(conf)
                        res_sample['Res'] = res_list
                        res_sample['Confidence'] = conf_list
                        res_sample['raw'] = raw_output
                    else:
                        res_text, conf = self.extract_res_and_confidence(raw_output)
                        res_sample['Res'] = res_text
                        res_sample['Confidence'] = conf
                        res_sample['raw'] = raw_output
                elif getattr(self.args, "using_topk_verb", False):
                    # Extract topk answer with max confidence, compatible with list raw_output
                    if self.args.using_api:
                        raw_output = qa_results[qa_ptr]
                    else:
                        raw_output = qa_results[qa_ptr][0]

                    if isinstance(raw_output, list):
                        # For list, extract topk max confidence answer for each item
                        res_list = []
                        conf_list = []
                        raw_list = []
                        for item in raw_output:
                            res_text, conf, raw = self.extract_topk_res_and_confidence(item)
                            res_list.append(res_text)
                            conf_list.append(conf)
                            raw_list.append(raw)
                        res_sample['Res'] = res_list
                        res_sample['Confidence'] = conf_list
                        res_sample['raw'] = raw_list
                    else:
                        res_text, conf, raw = self.extract_topk_res_and_confidence(raw_output)
                        res_sample['Res'] = res_text
                        res_sample['Confidence'] = conf
                        res_sample['raw'] = raw
                elif getattr(self.args, "using_api", False):
                    raw_output = qa_results[qa_ptr]
                    res_sample['Res'] = raw_output
                else:
                    raw_output = qa_results[qa_ptr]
                    if isinstance(raw_output, (tuple, list)) :
                        res_sample['Res'] = raw_output[0]
                        res_sample['logprobs_list'] = raw_output[1]
                        res_sample['token_ids'] = raw_output[2]

                res.append(res_sample)
                qa_ptr += 1
        logger.info(f'Processed data count: {qa_ptr}')

        out_dir = os.path.dirname(self.args.outfile)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        res_path = self.args.outfile
        if not res_path.endswith('.jsonl'):
            res_path = res_path + '.jsonl'

        res_path = self.get_outfile(res_path)

        import json

        print(f"Writing result file path: {res_path}")
        with open(res_path, 'w', encoding='utf-8') as f:
            for item in res:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return res

    def evaluate_res_llm(self, res=None):
        """
        Evaluate inference results, calculate has_answer and accuracy.
        res: Optional, if None then automatically read from outfile
        Return: res_with_eval, accuracy
        """
        import os
        acc = 0
        has_answer_list = []
        judging_res_list = []
        batch_size = getattr(self, "eval_model_name_bsz", 512)
        # Construct output path
        suffix = '_eval' 
        res_path = self.args.outfile
        res_path = self.get_outfile(res_path)
        res_path = res_path[:-6] + suffix + '.jsonl'
        metric_path = res_path[:-6] + '.metric'

        # Add: If exists and not require overwrite, read data and return only if exists (must have data to return)
        force_replace = self.args.force_replace
        if os.path.exists(res_path) and not force_replace:
            with open(res_path, "r", encoding="utf-8") as f:
                tmp_res = [json.loads(line) for line in f if line.strip()]
            if tmp_res:  # Return only if data exists (not empty)
                print(f"Evaluation result already exists: {res_path}, skipping evaluation.")
                acc_val = None
                if os.path.exists(metric_path):
                    with open(metric_path, "r", encoding="utf-8") as f:
                        metric = json.load(f)
                        acc_val = metric.get("accuracy", None)
                return tmp_res, acc_val

        qa_questions = [item['question'] for item in res]
        qa_references = [item['reference'] for item in res]
        qa_results = [item['Res'] for item in res]
        # Batch evaluation
        for i in range(0, len(res), batch_size):
            batch_questions = qa_questions[i:i+batch_size]
            batch_references = qa_references[i:i+batch_size]
            batch_results = qa_results[i:i+batch_size]
            if self.args.using_output_all:
                batch_has_answer, batch_judging_res = self.model_match_answer_batch_output_all(
                    batch_questions, batch_references, batch_results
                )
            else:
                batch_has_answer, batch_judging_res = self.model_match_answer_batch(
                    batch_questions, batch_references, batch_results
                )
            has_answer_list.extend(batch_has_answer)
            judging_res_list.extend(batch_judging_res)
        for i, item in enumerate(res):
            item['has_answer'] = has_answer_list[i]
            if item['has_answer'] is None:
                logger.warning('Failed to judge effectively:')
                logger.warning(judging_res_list[i])
            else:
                acc += int(item['has_answer'])

        logger.info(f'Evaluation data count: {len(res)}')
        logger.info(f'Accuracy: {acc / len(res) if len(res) else 0}')
        # Save evaluation results
        print(f"Writing evaluation result file path: {res_path}")
        with open(res_path, 'w', encoding='utf-8') as f:
            for item in res:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Writing metric file path: {metric_path}")
        with open(metric_path, 'w', encoding='utf-8') as f:
            json.dump({'accuracy': acc / len(res) if len(res) else 0, 'processed_data_count': len(res)}, f, ensure_ascii=False, indent=2)

        # If not self.using_topk and not self.using_vanilla, use calculated has_answer_list to supplement _eval_post.jsonl
        if not getattr(self, "using_topk", False) and not getattr(self, "using_vanilla", False):
            input_post_path = self.args.outfile.replace('.jsonl', '_post.jsonl')
            output_post_eval_path = self.args.outfile.replace('.jsonl', '_eval_post.jsonl')
            if os.path.exists(input_post_path):
                # Read original data
                items = []
                with open(input_post_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except Exception as ex:
                                logger.warning(f"json loads failed: {ex}")

                # Directly use the obtained has_answer_list
                for i, item in enumerate(items):
                    item['has_answer'] = has_answer_list[i] if i < len(has_answer_list) else None

                with open(output_post_eval_path, "w", encoding="utf-8") as fout:
                    for item in items:
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"Generated file with has_answer field: {output_post_eval_path}")

        # Similarly, merge/supplement has_answer results to _vanilla_verb.jsonl and _topk_verb.jsonl, output to _vanilla_verb_eval.jsonl and _topk_verb_eval.jsonl
        vanilla_verb_path = self.args.outfile.replace('.jsonl', '_vanilla_verb.jsonl')
        vanilla_verb_eval_path = self.args.outfile.replace('.jsonl', '_vanilla_verb_eval.jsonl')
        topk_verb_path = self.args.outfile.replace('.jsonl', '_topk_verb.jsonl')
        topk_verb_eval_path = self.args.outfile.replace('.jsonl', '_topk_verb_eval.jsonl')

        for inp_path, eval_path in [
            (vanilla_verb_path, vanilla_verb_eval_path),
            (topk_verb_path, topk_verb_eval_path)
        ]:
            if os.path.exists(inp_path):
                # Read original data
                items = []
                with open(inp_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except Exception as ex:
                                logger.warning(f"json loads failed in {inp_path}: {ex}")

                # Merge has_answer
                for i, item in enumerate(items):
                    item['has_answer'] = has_answer_list[i] if i < len(has_answer_list) else None

                with open(eval_path, "w", encoding="utf-8") as fout:
                    for item in items:
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"Generated file with has_answer field: {eval_path}")

        return res, acc / len(res) if len(res) else 0

    def load_data(self, data):
        self.data = data
        self.dataloader = DataLoader(self.data, shuffle=False, batch_size=self.batch_size)

    #      return has_answer_list, judging_res_list
    def model_match_answer_batch(self, questions, answers_list, states):
        """
        Batch judge if Res contains reference answer (reference can be one or multiple elements, Res is single answer)
        Submit for judgment separately for each reference
        """
        system_prompt = """
        You are given:
        - A single ground truth answer (reference)
        - A model-generated response (called `Res`)

        Task:
        Decide if `Res` explicitly contains the ground truth answer 
        or its clear variant.

        Matching rules:
        - Exact match, OR
        - Clear variant only (e.g., abbreviations, middle names, initials, hyphenated vs. non-hyphenated forms).
        - Generic refusals, disclaimers, or irrelevant text do NOT count as a match.

        Constraints:
        - Do NOT use external knowledge.
        - Do NOT assume correctness beyond literal comparison.
        - Only judge whether the given ground truth answer string (or its clear variant) 
        appears in `Res`.

        Output:
        - Respond strictly with "True" if a match is found.
        - Respond strictly with "False" if no match is found.
        """.strip()

        has_answer_list = []
        judging_res_list = []

        messages_batch = []
        index_map = []  # Record (sample idx, ref idx)

        # Iterate through each sample
        for i, (q, answers, s) in enumerate(zip(questions, answers_list, states)):
            # Initialize as undetermined
            has_answer_list.append(None)
            judging_res_list.append("Need model judgment")

            # If no reference, skip
            if not answers:
                continue

            # Generate query for each reference separately
            for ref in answers:
                content = get_evaluate_output_prompt(q, [ref], s, self.args)
                messages_batch.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ])
                index_map.append(i)

        if not messages_batch:
            return has_answer_list, judging_res_list
        
        # Batch submit to model
        if self.args.using_api:
            results = self.local_chat_batch_eval_api(messages_batch)
        elif self.args.using_host:
            results = self.local_chat_batch_eval_host(messages_batch, max_tokens=8)
        else:
            results = self.local_chat_batch_eval(messages_batch, max_tokens=8)

        # Iterate through results
        for idx, res in enumerate(results):
            sample_id = index_map[idx]

            # If sample already judged as True, skip
            if has_answer_list[sample_id] is True:
                continue

            if res is None:
                continue

            res_lower = res.lower()
            found = False

            # Check True
            for term in true_terms:
                if term.lower() in res_lower:
                    has_answer_list[sample_id] = True
                    judging_res_list[sample_id] = res
                    found = True
                    break

            # Check False (Only if not yet judged)
            if not found:
                for term in false_terms:
                    if term.lower() in res_lower:
                        if has_answer_list[sample_id] is None:
                            has_answer_list[sample_id] = False
                            judging_res_list[sample_id] = res
                        found = True
                        break

            # If model output is ambiguous (neither True nor False)
            if not found and has_answer_list[sample_id] is None:
                judging_res_list[sample_id] = res

        return has_answer_list, judging_res_list

    def model_match_answer_batch_output_all(self, questions, references_list, res_list):
        
        system_prompt = """
            You are a precise answer matcher.

            Given:
            - A reference answer (may contain multiple correct items, separated by |, comma, or list form)
            - A model-generated response (`Res`), which may include one or more candidate answers.

            Task:
            Determine if `Res` explicitly contains **all** reference answers or their clear textual variants.

            Matching rules:
            - Variants include abbreviations, hyphenation, middle names, or equivalent surface forms.
            - If every reference answer (or its variant) appears in `Res`, output "True".
            - If any reference answer is missing, output "False".
            - Ignore generic refusals, disclaimers, or irrelevant text.
            - Do NOT use external knowledge or reasoning beyond literal string matching.

            Output:
            Respond **strictly** with:
            - "True"  → all reference answers (or variants) found in `Res`
            - "False" → otherwise
            """.strip()

        has_answer_list = []
        judging_res_list = []

        messages_batch = []

        for q, reference, res in zip(questions, references_list, res_list):
            has_answer_list.append(None)
            judging_res_list.append("Need model judgment")
            # ref_str: Assemble as str regardless if reference is list or str
            if isinstance(reference, list):
                # Remove empty/None
                ref_items = [str(ref) for ref in reference]
                ref_str = " | ".join(ref_items)
            elif reference:
                ref_str = str(reference)
            else:
                ref_str = ""
            content = get_evaluate_output_prompt(q, [ref_str] if ref_str else [], res, self.args)
            messages_batch.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ])

        if not messages_batch:
            return has_answer_list, judging_res_list

        if self.args.entity == "math":
            # For math domain, judge if reference and res content meaning are equivalent, provided same quantity and all refs in res (res might be comma separated string)
            for i, (reference, res) in enumerate(zip(references_list, res_list)):
                # print("reference", reference)
                # Normalize reference to list of stripped strings
                ref_items = [str(item).strip() for item in reference if str(item).strip()]

                # Process res_items, if it's string and possibly comma (,) separated, convert to list
                if isinstance(res, list):
                    # Unpack all elements (avoid list containing comma separated strings), handle uniformly
                    joined = ",".join(str(item) for item in res if item is not None)
                    res_split = [item.strip() for item in joined.split(",") if item.strip()]
                    res_items = res_split
                elif res is None:
                    res_items = []
                else:
                    # If res is string, split by comma
                    res_items = [item.strip() for item in str(res).split(",") if item.strip()]

                # print("res:", res_items)
                # print("ref:", ref_items)
                # Judge same quantity and every ref item is in res
                if len(ref_items) == len(res_items) and all(r in res_items for r in ref_items):
                    has_answer_list[i] = True
                    judging_res_list[i] = "Equal number and all ref in res (math fast set matching)"
                else:
                    has_answer_list[i] = False
                    judging_res_list[i] = "Not equal (math fast set matching)"
            return has_answer_list, judging_res_list

        if self.args.using_api:
            results = self.local_chat_batch_eval_api(messages_batch)
        elif getattr(self.args, "using_host", False):
            results = self.local_chat_batch_eval_host(messages_batch, max_tokens=8)
        else:
            results = self.local_chat_batch_eval(messages_batch, max_tokens=8)

        for i, res in enumerate(results):
            if res is None:
                continue
            res_lower = res.lower()
            found = False
            # Check True
            for term in true_terms:
                if term.lower() in res_lower:
                    has_answer_list[i] = True
                    judging_res_list[i] = res
                    found = True
                    break
            # Check False
            if not found:
                for term in false_terms:
                    if term.lower() in res_lower:
                        if has_answer_list[i] is None:
                            has_answer_list[i] = False
                            judging_res_list[i] = res
                        found = True
                        break
            if not found and has_answer_list[i] is None:
                judging_res_list[i] = res
        return has_answer_list, judging_res_list

    def model_semantic_batch(self, batch_list_a: list[list[str]], batch_list_b: list[list[str]]) -> list[list[bool]]:
        """
        Support batch judgment of semantic equivalence for multiple text pairs.
        Input:
            batch_list_a: List[List[str]], each element is a group of text A to be judged (can be multiple pairs, batch==sample quantity)
            batch_list_b: List[List[str]], each element is a group of text B to be judged
        Output:
            List[List[bool]], list in each batch is equivalence judgment (True/False) for each pair
        """
        assert len(batch_list_a) == len(batch_list_b), "Two batch lists must have the same length"

        # --- For math domain entity=="math", go via value equivalence check ---
        if getattr(self.args, "entity", None) == "math":
            batch_judgments = []
            for sample_a, sample_b in zip(batch_list_a, batch_list_b):
                sub_judgments = []
                for a, b in zip(sample_a, sample_b):
                    # Perform string value equivalence check for each pair
                    judgement = str(a).strip() == str(b).strip()
                    sub_judgments.append(judgement)
                batch_judgments.append(sub_judgments)
            return batch_judgments

        system_prompt = """
        You are a precise semantic judge.

        Given two responses (ResA and ResB), determine whether they are semantically equivalent.
        
        Definition of 'semantically equivalent':
        - They express the same meaning or information, even if worded differently.
        - Minor stylistic or phrasing differences are acceptable.
        - If one is a refusal, disclaimer, or unrelated, they are NOT equivalent.

        Instructions:
        - Base your judgment only on the literal content of the two responses.
        - Do NOT use external knowledge.
        - Respond strictly with:
            "True"  → if they are semantically equivalent
            "False" → otherwise
        """.strip()

        # --- Flatten to flat pairs, collect index map ---
        all_a, all_b = [], []
        index_map = []  # [(batch_idx, pair_idx)]
        for i, (sample_a, sample_b) in enumerate(zip(batch_list_a, batch_list_b)):
            assert len(sample_a) == len(sample_b), f"Quantity of a, b in sample {i} is inconsistent"
            for j, (a, b) in enumerate(zip(sample_a, sample_b)):
                all_a.append(a)
                all_b.append(b)
                index_map.append((i, j))  # Record original position of this pair

        # Construct messages_batch
        messages_batch = []
        for a, b in zip(all_a, all_b):
            content = f"ResA: {a}\nResB: {b}\nAre ResA and ResB semantically equivalent?"
            messages_batch.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ])

        # Call LLM for batch judgment
        if self.args.using_api:
            results = self.local_chat_batch_eval_api(messages_batch)
        elif getattr(self.args, "using_host", False):
            results = self.local_chat_batch_eval_host(messages_batch, max_tokens=8)
        else:
            results = self.local_chat_batch_eval(messages_batch, max_tokens=8)

        judgments_flat = []
        for res in results:
            if res is None:
                judgments_flat.append(False)
                continue

            res_lower = res.strip().lower()
            found = False

            for term in true_terms:
                if term in res_lower:
                    judgments_flat.append(True)
                    found = True
                    break
            if not found:
                for term in false_terms:
                    if term in res_lower:
                        judgments_flat.append(False)
                        found = True
                        break
            if not found:
                judgments_flat.append(False)

        # Restore to batch structure
        batch_judgments = [[] for _ in batch_list_a]
        for idx, (batch_idx, pair_idx) in enumerate(index_map):
            # Ensure order in each sample matches input
            batch_judgments[batch_idx].append(judgments_flat[idx])

        return batch_judgments



    def process_res(self, outs):
        for res in outs:
            self.outputs.append({'Res': res})
   
    def get_outfile(self, outfile):
        if self.args.using_vanilla_verb:
            outfile = outfile.replace('.jsonl', '_vanilla_verb.jsonl')
        if self.args.using_topk_verb:
            outfile = outfile.replace('.jsonl', '_topk_verb.jsonl')
        if self.args.using_latent:
            outfile = outfile.replace('.jsonl', '_latent.jsonl')
        if self.args.using_sme:
            outfile = outfile.replace('.jsonl', '_sme.jsonl')
        if self.args.using_sample:
            outfile = outfile.replace('.jsonl', '_sample.jsonl')
        if self.args.consistency_origin:
            outfile = outfile.replace('.jsonl', '_consistency.jsonl')
        if self.args.consistency_origin_weight_vanilla:
            outfile = outfile.replace('.jsonl', '_consistency_weight_vanilla.jsonl')
        if self.args.consistency_origin_weight_topk:
            outfile = outfile.replace('.jsonl', '_consistency_weight_topk.jsonl')
        return outfile

    def compute_latent(self):
        import json
        import numpy as np
        import os
        from tqdm import tqdm

        base_path = getattr(self.args, "outfile", None)
        sample_path = base_path.replace(".jsonl", "_sample.jsonl")
        greedy_path = base_path.replace(".jsonl", "_eval.jsonl")

        outfile = self.get_outfile(greedy_path)
        # New logic: If output file exists, content is not empty, and force overwrite is not set, return directly
        force_replace = getattr(self, "force_replace", getattr(self.args, "force_replace", False))
        if os.path.exists(outfile) and not force_replace:
            with open(outfile, 'r', encoding='utf-8') as f:
                res = [json.loads(line) for line in f if line.strip()]
            if len(res) > 0:
                print(f"Calculated latency results exist and count is {len(res)}: {outfile}, skipping calculation.")
                return res

        # Read greedy answers
        with open(greedy_path, "r", encoding="utf-8") as f:
            greedy_data = [json.loads(line) for line in f if line.strip()]

        # Read sample data
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f) if sample_path.endswith(".json") else [json.loads(line) for line in f if line.strip()]

        res = []
        for idx, greedy_item in tqdm(enumerate(greedy_data), total=len(greedy_data), desc="Calc latent confidence"):
            # Get greedy output and input prompt
            greedy_logprobs = greedy_item.get("logprobs_list", None)
            greedy_token_ids = greedy_item.get("token_ids", None)

            # Corresponding sample
            sample_item = sample_data[idx] if isinstance(sample_data, list) else sample_data[str(idx)]
            sample_logprobs_list = sample_item.get("logprobs_list", [])
            sample_token_ids_list = sample_item.get("token_ids", [])

            # -- Step1: Greedy perplexity/entropy/length-norm-entropy --
            if greedy_logprobs is not None and greedy_token_ids is not None and len(greedy_token_ids) > 0:
                perp, pe, lne, _ = self.compute_metrics_from_logprobs(greedy_logprobs, greedy_token_ids)
            
            # -- Step2: Multi-sample PE/LNE --
            sample_pes = []
            sample_lnes = []
            for out_logprobs, out_token_ids in zip(sample_logprobs_list, sample_token_ids_list):
                if out_logprobs is not None and out_token_ids is not None and len(out_token_ids) > 0:
                    _, sample_pe, sample_lne, _ = self.compute_metrics_from_logprobs(out_logprobs, out_token_ids)
                    if sample_pe is not None:
                        sample_pes.append(sample_pe)
                    if sample_lne is not None:
                        sample_lnes.append(sample_lne)
            avg_pe = float(np.mean(sample_pes)) if sample_pes else None
            avg_lne = float(np.mean(sample_lnes)) if sample_lnes else None

            conf_perp = self.convert_to_conf(perp)
            conf_pe   = self.convert_to_conf(avg_pe)
            conf_lne  = self.convert_to_conf(avg_lne)

            # Write back
            greedy_item = self.del_log(greedy_item)
            
            greedy_item["perp"] = perp
            greedy_item["perp_conf"] = conf_perp
            greedy_item["pe"] = avg_pe
            greedy_item["pe_conf"] = conf_pe
            greedy_item["lne"] = avg_lne
            greedy_item["lne_conf"] = conf_lne
            res.append(greedy_item)

        # Save back to file
        with open(outfile, 'w', encoding='utf-8') as f:
            for item in res:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Writing new results with confidence to: {outfile }")
        return res

    # # prob add
    def compute_multi_answer_prob_add(self):
        Thres=self.args.prob_add_thres
        import math
        outfile = getattr(self.args, 'outfile', None)
        outfile = outfile.replace('.jsonl', '_eval_sme.jsonl')
        if not os.path.exists(outfile):
            print("this file not exist", outfile)
            return

        with open(outfile, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        # First calculate merged sum of each big_cluster separately, get all confidence values
        confidence_list = []
        for d in data:
            if "gpt" in self.args.model_name or "deepseek" in self.args.model_name:
                cluster_probs = d.get("cluster_probs", [])
            else:
                cluster_probs = d.get("cluster_res_probs", [])
                cluster_probs = [sum(a) for a in cluster_probs]
                
            big_cluster = [p for p in cluster_probs if p >= Thres]
            confidence = sum(big_cluster)
            confidence_list.append(confidence)
        
        # Global normalization of confidence to [0,1]
        if confidence_list:
            conf_min = min(confidence_list)
            conf_max = max(confidence_list)
        else:
            conf_min = conf_max = 0.0

        filtered_confidences = []
        for d, confidence in zip(data, confidence_list):
            if conf_max != conf_min:
                norm_conf = (confidence - conf_min) / (conf_max - conf_min)
            else:
                norm_conf = 0.0
            d["ma_prob_add_conf"] = norm_conf
            filtered_confidences.append(d)

        out_sme_filtered = outfile.replace('_eval_sme.jsonl', '_eval_ma_prob_add.jsonl')
        with open(out_sme_filtered, 'w', encoding='utf-8') as f:
            for item in filtered_confidences:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Merged clusters and wrote results: {out_sme_filtered}")
        return filtered_confidences


    def get_res(self):
        self.outputs = []
        outfile = getattr(self.args, 'outfile', None)
        outfile = self.get_outfile(outfile)
        total_num = self.dataloader.dataset.__len__()
        exist_data = []
        finished_num = 0
        skip_cal = False
        
        if outfile and os.path.exists(outfile):
            try:
                with open(outfile, 'r', encoding='utf-8') as f:
                    exist_data = [json.loads(line) for line in f if line.strip()]
                    valid_lines = [item for item in exist_data]
                    finished_num = len(valid_lines)
                    if finished_num == total_num:
                        logger.info(f"{outfile} exists and all Res fields are not empty, reading directly")
                        res = valid_lines
            except Exception as e:
                logger.warning(f"Error reading existing result file: {e}")
                
        if outfile and os.path.exists(outfile) and not getattr(self.args, "force_replace", False):
            skip_cal = True
        if self.args.eval_only or self.args.using_latent or self.args.using_sme or self.args.consistency_origin or self.args.consistency_origin_weight_vanilla or self.args.consistency_origin_weight_topk or self.args.using_post or self.args.using_multi_answer_prob_add:
            skip_cal = True

        if not skip_cal:
            for idx, batch in enumerate(self.dataloader):
                if self.args.model_type == 'llm':
                    if getattr(self.args, "using_sample", False):
                        if getattr(self.args, "using_api", False):
                            outs = self.process_batch_llm_sample_api(batch, temperature=1, top_p=1)
                        else:
                            outs = self.process_batch_llm_sample(batch, temperature=1, top_p=1)
                    elif getattr(self.args, "using_host", False):
                        outs = self.process_batch_llm_host(batch, temperature=0, top_p=1)
                    elif getattr(self.args, "using_api", False):
                        outs = self.process_batch_llm_api(batch, temperature=0, top_p=1)
                    else:
                        outs = self.process_batch_llm(batch, temperature=0, top_p=1)
                else:
                    outs = []
                self.process_res(outs)
            res = self.get_res_file()
        
        if self.args.eval_only:
            logger.info(f"Starting answer evaluation")
            self.evaluate_res_llm(res)
        elif self.args.using_latent:
            logger.info(f"Starting latent calculation")
            self.compute_latent()
        elif self.args.using_sme:
            logger.info(f"Starting sme calculation")
            self.compute_sme()
        elif self.args.consistency_origin:
            logger.info(f"Starting consistency_origin calculation")
            self.compute_consistency_with_eval()
        elif self.args.consistency_origin_weight_vanilla:
            logger.info(f"Starting consistency_origin_weight_vanilla calculation")
            self.compute_consistency_with_eval_weight("vanilla_verb")
        elif self.args.consistency_origin_weight_topk:
            logger.info(f"Starting consistency_origin_weight_topk calculation")
            self.compute_consistency_with_eval_weight("topk_verb")
        elif self.args.using_multi_answer_prob_add:
            logger.info(f"Starting multi_answer_prob_add calculation")
            self.compute_multi_answer_prob_add()
        elif self.args.using_multi_answer_plogp:
            logger.info(f"Starting multi_answer_plogp calculation")
            self.compute_multi_answer_plogp()
        elif self.args.using_post:
            logger.info(f"Starting using_post calculation")   
            self.compute_post( 
                use_post_self=True,
                use_post_self_candidates=True,
                use_post_p_true=True,
                use_post_p_true_candidates=True,
                use_post_p_true_logits=True if not self.args.using_api else False,
                use_post_p_true_logits_candidates=True if not self.args.using_api else False,
                num_samples=self.args.sample_num
            )
        else:
            pass
        return 
    

    def compute_consistency_with_eval(
        self
    ):
        import os
        outfile = getattr(self.args, "outfile", None)
        sample_path = outfile.replace(".jsonl", "_sample.jsonl")
        greedy_path = outfile.replace(".jsonl", "_eval.jsonl")

        # === New logic: If output file exists, corresponding data is not empty, and force overwrite is not set, return directly ===
        rawanswer_output_path = greedy_path
        rawanswer_output_path = self.get_outfile(rawanswer_output_path)
        force_replace = getattr(self, "force_replace", getattr(self.args, "force_replace", False))
        if os.path.exists(rawanswer_output_path) and not force_replace:
            # Check if file content is non-empty (at least one valid json line)
            try:
                with open(rawanswer_output_path, "r", encoding="utf-8") as fin:
                    has_data = any(line.strip() for line in fin)
            except Exception as ex:
                has_data = False
            if has_data:
                logger.info(f"consistency(confidence) output exists and is not empty: {rawanswer_output_path}, skipping.")
                return

        # Read greedy answers
        with open(greedy_path, "r", encoding="utf-8") as f:
            greedy_data = [json.loads(line) for line in f if line.strip()]

        # Read sample data
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f) if sample_path.endswith(".json") else [json.loads(line) for line in f if line.strip()]

        # === Batch processing (Acceleration: Overall batching) ===
        # 1. Collect all samples to be processed first
        batch_questions = []
        batch_references = []
        batch_states = []
        batch_indices = []  # Record which row in greedy_data each sample belongs to
        batch_counts = []   # Record quantity of multi_answers for each sample, for later splitting

        for i in range(len(greedy_data)):
            greedy_res = greedy_data[i]["Res"]
            multi_answers = sample_data[i]["Res"]
            greedy_data[i]["sample"] = multi_answers

            if not multi_answers:
                greedy_data[i]["confidence"] = 0.0
                batch_counts.append(0)
                continue

            questions = ["" for _ in multi_answers]

            references = []
            for a in multi_answers:
                if isinstance(a, list):
                    references.append(a)
                elif isinstance(a, str) and ',' in a:
                    references.append([item.strip() for item in a.split(',') if item.strip()])
                else:
                    references.append([a])
            states = [greedy_res] * len(multi_answers)

            batch_questions.extend(questions)
            batch_references.extend(references)
            batch_states.extend(states)
            batch_indices.append(i)
            batch_counts.append(len(multi_answers))

        # 2. Batch call model
        if batch_questions:
            if self.args.using_output_all:
                has_answer_list, _ = self.model_match_answer_batch_output_all(
                    batch_questions, batch_references, batch_states
                )
            else:
                has_answer_list, _ = self.model_match_answer_batch(
                    batch_questions, batch_references, batch_states
                )
        else:
            has_answer_list = []

        # 3. Split results according to multi_answers count of each greedy_data, write confidence
        ptr = 0
        for idx, count in zip(batch_indices, batch_counts):
            if count == 0:
                continue
            sub_has_answer = has_answer_list[ptr:ptr+count]
            match_count = sum(1 for x in sub_has_answer if x is True)
            total = count
            consistency_value = match_count / total if total > 0 else 0.0
            greedy_data[idx] = self.del_log(greedy_data[idx])
            greedy_data[idx]["consistency_conf"] = consistency_value
            ptr += count

        # === Write back to file ===
        write_jsonl(greedy_data, rawanswer_output_path)
        logger.info(f"✅ Written consistency(confidence field): {rawanswer_output_path}")


    
    def compute_consistency_with_eval_weight(
        self,
        conf_suffix : str
    ):
        """
        This function mimics compute_consistency_with_eval, calculating final reliability using weighted method (summing confidence weights) for each multi_answer
        New logic: If output file exists and not force overwrite, return directly
        """
        from utils.utils import read_json, write_jsonl
        import os

        outfile = getattr(self.args, "outfile", None)
        # Construct filename following compute_consistency_with_eval logic
        sample_path = outfile.replace(".jsonl", f"_{conf_suffix}_sample.jsonl")
        greedy_path = outfile.replace(".jsonl", "_eval.jsonl")

        # ==== New logic: Check if output file exists and is not empty ====
        rawanswer_output_path = greedy_path
        rawanswer_output_path = self.get_outfile(rawanswer_output_path)
        force_overwrite = getattr(self.args, 'force_replace', False)
        output_exists = os.path.exists(rawanswer_output_path)
        output_nonempty = False
        if output_exists and not force_overwrite:
            try:
                with open(rawanswer_output_path, "r", encoding="utf-8") as f:
                    # Check if file has non-empty lines (valid data)
                    for line in f:
                        line = line.strip()
                        if line and line not in ("[]", "{}", ""):
                            output_nonempty = True
                            break
            except Exception as e:
                logger.warning(f"⚠️ Error checking output file content: {e}")
        if output_exists and output_nonempty:
            logger.info(f"⚠️ Output file exists and data is not empty, force overwrite not set, skipping write: {rawanswer_output_path}")
            return

        # Read greedy original answers (already evaluated standard model output)
        with open(greedy_path, "r", encoding="utf-8") as f:
            greedy_data = [json.loads(line) for line in f if line.strip()]

        # Read sample results corresponding to conf_suffix (with Confidence field)
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_data = [json.loads(line) for line in f if line.strip()]

        if len(greedy_data) != len(sample_data):
            logger.warning(
                f"⚠️ Quantity mismatch: greedy_data={len(greedy_data)} vs sample_data={len(sample_data)}"
            )
            return

        # === Batch processing ===
        batch_questions = []
        batch_references = []
        batch_states = []
        batch_indices = []  # Record which row in greedy_data/sample_data each sample belongs to
        batch_counts = []   # Record quantity of multi_answers for each sample, for later splitting

        for i in range(len(greedy_data)):
            greedy_res = greedy_data[i].get("Res", "")
            multi_answers = sample_data[i].get("Res", [])
            confidences = sample_data[i].get("Confidence", [])

            greedy_data[i]["sample"] = multi_answers
            greedy_data[i]["sample_confidences"] = confidences

            if not multi_answers or not confidences or len(multi_answers) != len(confidences):
                greedy_data[i]["confidence"] = 0.0
                batch_counts.append(0)
                continue

            questions = ["" for _ in multi_answers]
            references = [[a] if not isinstance(a, list) else a for a in multi_answers]
            states = [greedy_res] * len(multi_answers)

            batch_questions.extend(questions)
            batch_references.extend(references)
            batch_states.extend(states)
            batch_indices.append(i)
            batch_counts.append(len(multi_answers))

        # 2. Batch call model
        if batch_questions:
            if self.args.using_output_all:
                has_answer_list, _ = self.model_match_answer_batch_output_all(    
                batch_questions, batch_references, batch_states
            )
            else:
                has_answer_list, _ = self.model_match_answer_batch(
                    batch_questions, batch_references, batch_states
                )
        else:
            has_answer_list = []

        # 3. Split results according to multi_answers count of each greedy_data, write weighted consistency
        ptr = 0
        for idx, count in zip(batch_indices, batch_counts):
            if count == 0:
                continue
            sub_has_answer = has_answer_list[ptr:ptr+count]
            confidences = sample_data[idx].get("Confidence", [])
            if len(confidences) != count:
                # Length anomaly, set to 0
                greedy_data[idx]["confidence"] = 0.0
                ptr += count
                continue

            weighted_sum = 0.0
            conf_sum = 0.0
            for match, ci in zip(sub_has_answer, confidences):
                try:
                    ci_float = float(ci)
                except Exception:
                    ci_float = 0.0
                if match is True:
                    weighted_sum += ci_float
                conf_sum += ci_float
            confidence = weighted_sum / conf_sum if conf_sum > 0 else 0.0
            greedy_data[idx] = self.del_log(greedy_data[idx])
            greedy_data[idx]["consistency_conf"] = confidence
            ptr += count

        # === Write back to file ===
        write_jsonl(greedy_data, rawanswer_output_path)
        logger.info(f"✅ Written consistency(confidence field): {rawanswer_output_path}")

    # def process_batch_llm(self, batch, temperature=0, top_p=1.0):
    #     if self.args.multi_step_type is None:
    #         new_batch = batch
    #     else:
    #         new_batch = batch['question']
    #     all_messages = []
    #     for idx, input_data in enumerate(new_batch):
    #         messages = [
    #             {'role': 'system', 'content': 'You are a helpful assistant.'},
    #             {'role': 'user', 'content': input_data}
    #         ]
    #         all_messages.append(messages)
    #     responses = self.local_chat_batch(all_messages, temperature, top_p)
    #     return responses

    def process_batch_llm(self, batch, temperature=1, top_p = 1):
        if self.args.multi_step_type is None:
            new_batch = batch
        else:
            new_batch = batch['question']

        all_messages = []
        for idx, input_data in enumerate(new_batch):
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': input_data}
            ]
            all_messages.append(messages)

        prompts = [format_messages_to_prompt(messages, self.model_name) for messages in all_messages]

        sampling_params_single = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            max_tokens=512,
            logprobs=20
        )
        outputs_single = self.llm.generate(prompts, sampling_params_single)

        results = []
        # Iterate through each prompt in batch
        for b_idx, prompt in enumerate(prompts):
            if outputs_single[b_idx].outputs:
                text = outputs_single[b_idx].outputs[0].text.strip()
                logprobs_raw = outputs_single[b_idx].outputs[0].logprobs
                token_ids = outputs_single[b_idx].outputs[0].token_ids

                # ---- New: Extract token_id: logprob ----
                clean_logprobs_list = []
                for step_dict in logprobs_raw:  # logprobs for each step is a dictionary
                    clean_step = {}
                    for token_id, logprob_obj in step_dict.items():
                        try:
                            clean_step[token_id] = logprob_obj.logprob  # Extract value
                        except AttributeError:
                            # If already float or string, keep directly
                            clean_step[token_id] = float(logprob_obj) if isinstance(logprob_obj, (int, float)) else str(logprob_obj)
                    clean_logprobs_list.append(clean_step)

        # ---- Save results ----
        results.append((text, clean_logprobs_list, token_ids))

        return results

    import requests
    import json

    def process_batch_llm_host(self, batch, temperature=1, top_p=1):
        """
        Use local vLLM port (http://localhost:8000/v1/completions)
        Perform inference, return (text, clean_logprobs_list, token_ids)
        Support batch request acceleration, and show progress bar
        """
        import requests
        from tqdm import tqdm

        if self.args.multi_step_type is None:
            new_batch = batch
        else:
            new_batch = batch['question']

        # Construct prompt list
        prompts = []
        for idx, input_data in enumerate(new_batch):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_data}
            ]
            # Convert to standard text prompt
            prompt_text = format_messages_to_prompt(messages, self.model_name)
            prompts.append(prompt_text)

        results = []

        # Support super large batch splitting
        max_api_batch = 512  # Local vllm api recommends around 16, can increase based on VRAM
        total = len(prompts)
        batched_prompts = [
            prompts[i:i + max_api_batch] for i in range(0, len(prompts), max_api_batch)
        ]

        pbar = tqdm(total=total, desc="vLLM API batch infer (host)", ncols=80)

        for batch_prompts in batched_prompts:
            batch_results = self.process_batch_llm_host_prompts_log(batch_prompts, temperature, top_p)
            results.extend(batch_results)

        pbar.close()
        return results

    
    def process_batch_llm_api(
        self,
        batch,
        api_key: str = "sk-tTo3MNJgAsRIvFgyuRCWfUKSVkBpIgBtPZi7yKTGGAmspl5D",
        api_url: str = "http://123.129.219.111:3000/v1/chat/completions",
        model: str = "gpt-4o-mini",
        concurrency: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_retries: int = 3,
    ):
        model = self.model_name
        import nest_asyncio
        nest_asyncio.apply()

        if self.args.multi_step_type is None:
            new_batch = batch
        else:
            new_batch = batch["question"]
        # print("api_url type:", type(api_url), "value:", api_url)
        # Construct message list
        messages_list = []
        for input_data in new_batch:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_data},
            ]
            # print(type(input_data))
            messages_list.append(messages)

        async def async_batch_infer():
            semaphore = asyncio.Semaphore(concurrency)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "BatchEvalBot/1.0",
            }
            # print(f"headers sample: {headers}")

            async def call_api(session, messages):
                """Single request logic (with retry + error printing)"""
                for attempt in range(max_retries):
                    try:
                        payload = {
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                        # print(f"Payload sample: {payload}")
                        async with semaphore:
                            async with session.post(
                                api_url,
                                json=payload,
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=90),
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    return data.get("choices", [{}])[0].get("message", {}).get("content", None)
                                else:
                                    text = await resp.text()
                                    print(f"⚠️ HTTP {resp.status} - attempt {attempt+1}: {text}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep((2 ** attempt) + random.random())
                    except Exception as e:
                        print(f"❌ Exception in call_api (attempt {attempt+1}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep((2 ** attempt) + random.random())
                return None


            async with aiohttp.ClientSession() as session:
                tasks = [call_api(session, msg) for msg in messages_list]
                results = await tqdm_asyncio.gather(*tasks, desc="Remote LLM API batch inference")
                return results

        # Sync wrap
        results = asyncio.run(async_batch_infer())
        return results

    def process_batch_llm_host_prompts_log(self, prompts, temperature=0, top_p=1):
        
        import requests
        from tqdm import tqdm
        # ---- Define API request info ----
        api_url = "http://localhost:8000/v1/completions"
        headers = {"Content-Type": "application/json"}

        results = []

        # Support super large batch splitting
        max_api_batch = 512  # Local vllm api recommends around 16, can increase based on VRAM
        total = len(prompts)
        batched_prompts = [
            prompts[i:i + max_api_batch] for i in range(0, len(prompts), max_api_batch)
        ]

        pbar = tqdm(total=total, desc="vLLM API batch infer (host)", ncols=80)

        for batch_prompts in batched_prompts:
            payload = {
                "model": self.model_path,
                "prompt": batch_prompts,   # Send multiple prompts at once
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 512,
                "logprobs": 20,     # Request top 20 logprobs for each token
                "stream": False
            }
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=180)
            except Exception as e:
                print(f"❌ Batch request failed: {e}")
                # Fill empty slots to maintain correspondence with input
                for _ in batch_prompts:
                    results.append(("", [], []))
                    pbar.update(1)
                continue

            if response.status_code != 200:
                print(f"❌ Status code {response.status_code}, response content: {response.text}")
                # Fill empty slots
                for _ in batch_prompts:
                    results.append(("", [], []))
                    pbar.update(1)
                continue

            data = response.json()
            # data["choices"] is a list when batch returning, corresponding to each prompt
            choices = data.get("choices", [])
            for choice in choices:
                text = choice.get("text", "").strip()
                # vLLM format: logprobs → { "tokens": [...], "token_logprobs": [...], "top_logprobs": [...] }
                logprobs_raw = choice.get("logprobs", {}).get("top_logprobs", [])
                token_ids_raw = choice.get("logprobs", {}).get("tokens", []) or []  # Actually decoded tokens

                # ---- 🔧 Clean logprobs ----
                clean_logprobs_list = []
                for step_dict in logprobs_raw:
                    clean_step = {}
                    if not isinstance(step_dict, dict):
                        continue
                    for token_str, logprob_obj in step_dict.items():
                        # Convert key to str
                        token_key = str(token_str)
                        try:
                            if isinstance(logprob_obj, dict) and "logprob" in logprob_obj:
                                clean_step[token_key] = float(logprob_obj["logprob"])
                            else:
                                clean_step[token_key] = float(logprob_obj)
                        except Exception:
                            clean_step[token_key] = str(logprob_obj)
                    clean_logprobs_list.append(clean_step)

                # ---- 🔧 Unify token_ids to string list ----
                tokens = [str(tok) for tok in token_ids_raw]

                results.append((text, clean_logprobs_list, tokens))
                pbar.update(1)

        pbar.close()
        return results

    def del_log(self, greedy_item):
        if "logprobs_list" in greedy_item:
            del greedy_item["logprobs_list"]
        if "token_ids" in greedy_item:
            del greedy_item["token_ids"]
        return greedy_item

    def compute_sme(self):
        """
        SME calculation (Semantic Entropy)
        Adapts to new version model_semantic_batch(batch_list_a, batch_list_b),
        where all pair(str) pairs within each sample are stored independently.
        When saving, also keep overall probability of each cluster, and probability of each res in each cluster.
        Added logic: If strings a and b are exactly equal, consider semantically equivalent directly, no need to pass to model judgment.
        """
        import json
        import numpy as np
        import os

        greedy_path = getattr(self.args, "outfile", None)
        sample_path = greedy_path.replace(".jsonl", "_sample.jsonl")
        greedy_path = greedy_path.replace(".jsonl", "_eval.jsonl")
        outfile = greedy_path.replace(".jsonl", "_sme.jsonl")

        # New logic: If output file exists, corresponding data is not empty, and force overwrite is not set, return directly
        force_replace = getattr(self, "force_replace", getattr(self.args, "force_replace", False))
        if os.path.exists(outfile) and not force_replace:
            with open(outfile, 'r', encoding='utf-8') as f:
                res = [json.loads(line) for line in f if line.strip()]
            if len(res) > 0:
                print(f"SME results exist and file is not empty: {outfile}, skipping calculation.")
                return res

        # --- Read data ---
        with open(greedy_path, "r", encoding="utf-8") as f:
            greedy_data = [json.loads(line) for line in f if line.strip()]

        with open(sample_path, "r", encoding="utf-8") as f:
            if sample_path.endswith(".json"):
                sample_data = json.load(f)
            else:
                sample_data = [json.loads(line) for line in f if line.strip()]

        # --- Collect pairs within each sample, and record which pairs are exactly equal ---
        batch_list_a, batch_list_b = [] , []
        pair_indices_per_sample = []
        equal_pairs_per_sample = []

        for idx, greedy_item in enumerate(greedy_data):
            sample_item = sample_data[idx] if isinstance(sample_data, list) else sample_data[str(idx)]
            seqs = sample_item.get("Res", None)
            if seqs is None:
                pair_indices_per_sample.append([])
                batch_list_a.append([])
                batch_list_b.append([])
                equal_pairs_per_sample.append(set())
                continue

            M_now = len(seqs)
            pair_a, pair_b, pair_indices = [], [], []
            equal_pairs = set()
            for m in range(M_now):
                for n in range(m + 1, M_now):
                    a_str = str(seqs[m])
                    b_str = str(seqs[n])
                    if a_str == b_str:
                        # Record index pair for equality case
                        equal_pairs.add((m, n))
                    else:
                        pair_a.append(a_str)
                        pair_b.append(b_str)
                        pair_indices.append((m, n))

            batch_list_a.append(pair_a)
            batch_list_b.append(pair_b)
            pair_indices_per_sample.append(pair_indices)
            equal_pairs_per_sample.append(equal_pairs)

        # --- Batch semantic judgment only for non-equal pairs ---
        all_batch_judgments = self.model_semantic_batch(batch_list_a, batch_list_b)

        # --- Calculate SME per sample ---
        for idx, greedy_item in enumerate(greedy_data):
            sample_item = sample_data[idx] if isinstance(sample_data, list) else sample_data[str(idx)]
            seqs = sample_item.get("Res", None)
            logprobs_list_list = sample_item.get("logprobs_list", [])
            token_ids_list = sample_item.get("token_ids", [])

            # Judge if logprobs_list_list and token_ids_list are missing
            use_cluster_size_probability = (
                (logprobs_list_list is None or token_ids_list is None) or
                not isinstance(logprobs_list_list, list) or
                not isinstance(token_ids_list, list) or
                len(logprobs_list_list) == 0 or
                len(token_ids_list) == 0
            )

            probs, seq_logps = [], []
            if not use_cluster_size_probability:
                for logprobs_seq, token_ids_seq in zip(logprobs_list_list, token_ids_list):
                    _, _, _, seq_logp = self.compute_metrics_from_logprobs(logprobs_seq, token_ids_seq)
                    seq_logps.append(seq_logp)
                    probs.append(float(np.exp(seq_logp)))

            # If no sequence, assign 0 directly
            if seqs is None:
                greedy_item["sme_conf"] = 0.0
                greedy_item["cluster_probs"] = []
                greedy_item["cluster_res_probs"] = []
                continue

            M_now = len(seqs)
            pair_indices = pair_indices_per_sample[idx]
            equal_pairs = equal_pairs_per_sample[idx]
            # judgments corresponds only to non-equal pairs
            judgments = all_batch_judgments[idx] if idx < len(all_batch_judgments) else []
            pair_to_equiv = {}

            # First fill True for equal pairs
            for eq_pair in equal_pairs:
                pair_to_equiv[eq_pair] = True
            # Then fill equivalence judgment results for non-equal pairs
            for k in range(len(judgments)):
                pair_to_equiv[pair_indices[k]] = judgments[k]

            # --- Clustering ---
            clusters = []
            used = set()
            for m in range(M_now):
                if m in used:
                    continue
                c = [m]
                for n in range(m + 1, M_now):
                    if n in used:
                        continue
                    if pair_to_equiv.get((m, n), False):
                        c.append(n)
                        used.add(n)
                clusters.append(c)

            # --- Entropy calculation, and get probabilities per cluster ---
            cluster_sum_probs = []
            cluster_res_probs = []
            if M_now > 0:
                if not use_cluster_size_probability:
                    # Sum of probabilities for each cluster
                    p_c = np.array([np.sum([probs[i] for i in c]) for c in clusters])
                    # Probability of res in each cluster
                    cluster_res_probs = [[probs[i] for i in c] for c in clusters]
                    # Normalize probability for each cluster
                    if p_c.sum() > 0:
                        p_c_norm = p_c / p_c.sum()
                        se = -np.sum(p_c_norm * np.log(p_c_norm + 1e-12))
                        conf_se = float(np.exp(-se))
                    else:
                        p_c_norm = np.zeros_like(p_c)
                        conf_se = 0.0
                    # Keep probabilities for each cluster (normalized)
                    cluster_sum_probs = p_c_norm.tolist()
                else:
                    # If no logprobs, use cluster size / total sample count to get cluster probability
                    cluster_sizes = np.array([len(c) for c in clusters])
                    total_size = np.sum(cluster_sizes)
                    if total_size > 0:
                        p_c_norm = cluster_sizes / total_size
                        se = -np.sum(p_c_norm * np.log(p_c_norm + 1e-12))
                        conf_se = float(np.exp(-se))
                    else:
                        p_c_norm = np.zeros_like(cluster_sizes)
                        conf_se = 0.0
                    cluster_sum_probs = p_c_norm.tolist()
                    cluster_res_probs = [[1.0/len(c)]*len(c) if len(c) > 0 else [] for c in clusters]
            else:
                cluster_sum_probs = []
                cluster_res_probs = []
                conf_se = 0.0

            # Clean large fields no longer needed
            greedy_item = self.del_log(greedy_item)

            greedy_item["seqs"] = seqs
            greedy_item["cluster"] = clusters
            greedy_item["sme_conf"] = conf_se   # Keep two decimal places
            greedy_item["cluster_probs"] = cluster_sum_probs
            greedy_item["cluster_res_probs"] = cluster_res_probs

        # --- Write back to file ---
        with open(outfile, 'w', encoding='utf-8') as f:
            for item in greedy_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✅ Written SME confidence field (sme_conf) and cluster_probs, cluster_res_probs: {outfile}")
        return greedy_data
  


    def model_semantic_single(self, a: str, b: str) -> bool:
        """
        Judge if two answers are semantically equivalent only when needed.
        Can be changed to async queue version or API batch processing.
        """
        # Pruning: Exactly same means True directly
        if a.strip() == b.strip():
            return True

        system_prompt = (
            "You are a precise semantic judge.\n"
            "Given two responses (ResA and ResB), determine whether they are semantically equivalent.\n"
            "Respond with True or False only."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ResA: {a}\nResB: {b}\nAre they semantically equivalent?"}
        ]

        try:
            res = self.local_chat_batch_eval_api([messages])[0]
            if not res:
                return False
            res = res.strip().lower()
            if any(x in res for x in ["true", "yes"]):
                return True
            elif any(x in res for x in ["false", "no"]):
                return False
            else:
                return False
        except Exception as e:
            print(f"❌ Judgment failed: {e}")
            return False




    def compute_metrics_from_logprobs(self, logprobs_list, token_ids=None):
        """
        Only supports single logprobs_list input:
        - logprobs_list: List[dict], each element is a logprobs dictionary for a token position (key: str token, value: log_prob)
        - token_ids: List[str], corresponding generated token string sequence
        """
        import math
        import numpy as np

        seq_logp = 0.0
        seq_len = 0

        token_logps = []

        for idx, lp in enumerate(logprobs_list):
            if lp is None:
                continue

            chosen_logprob = None

            # lp: {token (str): log_prob (float)}

            if token_ids is not None:
                token = token_ids[idx]
                if token in lp:
                    chosen_logprob = lp[token]

            if chosen_logprob is not None:
                seq_logp += chosen_logprob
                token_logps.append(chosen_logprob * math.exp(chosen_logprob))  # p * logp
                seq_len += 1

        # ===== Calculate three metrics =====
        # 1. Perplexity / Length-Normalized Entropy
        if seq_len > 0:
            perp = -seq_logp / seq_len
        else:
            perp = None

        # 2. Predictive Entropy   -Sum plogp 
        if seq_len > 0:
            pe = -sum(token_logps)
        else:
            pe = None

        if seq_len > 0:
            lne = -sum(token_logps) / seq_len
        else:
            lne = None

        return perp, pe, lne, seq_logp




    def convert_to_conf(self, metric):
        import math
        conf = math.exp(-metric)
        return conf

    def process_batch_llm_sample(self, batch, temperature, top_p):
        repeated_batch = []
        for input_data in batch:
            repeated_batch.extend([input_data] * self.args.sample_num)
        if self.args.using_host:
            results = self.process_batch_llm_host(repeated_batch, temperature, top_p)
        else:
            results = self.process_batch_llm(repeated_batch, temperature, top_p)

        grouped_results = []
        for i in range(0, len(results), self.args.sample_num):
            res_list = []
            logprobs_list_list = []
            token_ids_list = []
            group = results[i:i + self.args.sample_num]
            for item in group:
                res, logprobs, token_ids = item
                res_list.append(res)
                logprobs_list_list.append(logprobs)
                token_ids_list.append(token_ids)
            grouped_results.append((res_list, logprobs_list_list, token_ids_list))
        return grouped_results

    def process_batch_llm_sample_api(self, batch, temperature, top_p):
        repeated_batch = []
        for input_data in batch:
            repeated_batch.extend([input_data] * self.args.sample_num)

        # print(repeated_batch)
        results = self.process_batch_llm_api(repeated_batch, temperature=temperature, top_p=top_p)
        
        grouped_results = []
        for i in range(0, len(results), self.args.sample_num):
            res_list = []
            group = results[i:i + self.args.sample_num]
            for item in group:
                res_list.append(item)
            grouped_results.append(res_list)

        # print(grouped_results)
        return grouped_results

    @staticmethod
    def _extract_first_float_batch(texts: List[str]) -> List[Optional[float]]:
        results = []
        for text in texts:
            if text is None:
                results.append(None)
                continue
            m = re.search(r'[-+]?\d*\.?\d+', text)
            if not m:
                results.append(None)
            else:
                try:
                    results.append(float(m.group(0)))
                except Exception:
                    results.append(None)
        return results

    @staticmethod
    def _round_batch(xs: List[Optional[float]]) -> List[Optional[float]]:
        """
        Round a list of floats to two decimal places.
        Keep None values as is.
        """
        rounded = []
        for x in xs:
            if x is None:
                rounded.append(None)
            else:
                rounded.append(round(x, 2))
        return rounded

    @staticmethod
    def _parse_bool_text_batch(texts: List[str]) -> List[Optional[bool]]:
        results = []
        for s in texts:
            if not s:
                results.append(None)
                continue
            t = s.strip().lower()
            if re.search(r'\btrue\b', t):
                results.append(True)
            elif re.search(r'\bfalse\b', t):
                results.append(False)
            else:
                results.append(None)
        return results

    # ========= prompt construction =========
    def _self_conf_messages(self, question: str, answer: str):
        system = (
            "You are a careful fact-checker. "
            "You will be given a Question and an Answer. "
            "Your task is to estimate the probability (0.00 to 1.00) that the Answer is factually correct "
            "for the Question, based ONLY on your internal knowledge. "
            "Output a single number between 0.00 and 1.00, rounded to TWO decimal places. "
            "Output ONLY the number, no explanation or extra text."
        )
        user = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Respond ONLY with the probability (0.00–1.00, two decimal places)."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _self_conf_candidates_messages(self, question: str, answer: str, candidates: Optional[List[str]] = None):
        system = (
            "You are a careful fact-checker. "
            "You will be given a Question, some candidate possible answers, and one candidate Answer. "
            "Your task is to estimate the probability (0.00 to 1.00) that the candidate Answer is factually correct "
            "for the Question, based ONLY on your internal knowledge. "
            "Output a single number between 0.00 and 1.00, rounded to TWO decimal places. "
            "Output ONLY the number, no explanation or extra text."
        )
        candidate_str = ", ".join(candidates) if candidates else "None"
        user = (
            f"Question: {question}\n"
            f"Candidate possible answers: {candidate_str}\n"
            f"Candidate Answer: {answer}\n"
            f"Respond ONLY with the probability (0.00–1.00, two decimal places)."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


    def _p_true_judge_messages(self, question: str, answer: str):
        system = (
        "You are a careful fact-checker. "
        "You will be given a Question and one Possible Answer. "
        "Your task is to decide if the Possible Answer is factually correct for the Question. "
        "Output only 'True' or 'False'. Do not explain."
        )
        
        # Repair plan 1: Directly request judgment
        user = (
            f"Question: {question}\n"
            f"Possible Answer: {answer}\n"
            f"Is the possible answer factually correct? Answer only 'True' or 'False':"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _p_true_candidates_judge_messages(self, question:str, answer: str, candidates:list) :
        system = (
            "You are a careful fact-checker. "
            "You will be given a Question, some candidatesed candidate answers, "
            "and one Possible Answer. "
            "Your task is to decide if the Possible Answer is factually correct for the Question. "
            "Output only 'True' or 'False'. Do not explain."
            )
        candidate_str = ", ".join(candidates)
        user = (
            f"Question: {question}\n"
            f"Here are some candidatesed ideas: {candidate_str}\n"
            f"Possible Answer: {answer}\n"
            f"The possible answer is:"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ========= Confidence Calculation =========
    def confidence_from_self_report_batch(
        self,
        questions: List[str],
        answers: List[str],
        max_tokens: int = 8,
        candidates_list = None
    ):
        if candidates_list is not None:
            messages_list = []
            for idx, (q, a) in enumerate(zip(questions, answers)):
                candidates = candidates_list[idx]
                messages_list.append(self._self_conf_candidates_messages(q, a, candidates))
        else:
            messages_list = [self._self_conf_messages(q, a) for q, a in zip(questions, answers)]
        if self.args.using_host:
            outs = self.local_chat_batch_host(messages_list, temperature=0.0, top_p=1.0, max_tokens=max_tokens)
        elif self.args.using_api:
            outs = self.local_chat_batch_eval_api(messages_list)
        else:
            outs = self.local_chat_batch(messages_list, temperature=0.0, top_p=1.0, max_tokens=max_tokens)

        float_vals = self._extract_first_float_batch(outs)
        rounded_vals = float_vals
        
        result_list = []
        for rounded_val, raw_val in zip(rounded_vals, outs):
            if candidates_list is None:
                result_list.append({"self_report_raw": raw_val,"self_report_conf": rounded_val, })
            else:
                result_list.append({"self_report_candidates_raw": raw_val,"self_report_candidates_conf": rounded_val})
        return result_list


    def confidence_from_p_true_logits_batch(
        self,
        questions: List[str],
        answers: List[str],
        candidates_list: Optional[List[Optional[List[str]]]] = None,
    ) -> List[Dict]:
        """
        Get True/False logits probability.
        If self.args.using_host = True, infer via local port 8000 vLLM service.
        Otherwise use self.llm.generate.
        """
        import math
        import requests
        import json

        results = []

        # === Construct prompts ===
        if candidates_list is None:
            prompts = [
                format_messages_to_prompt(
                    self._p_true_judge_messages(q, a),
                    self.model_name
                )
                for q, a in zip(questions, answers)
            ]
        else:
            prompts = [
                format_messages_to_prompt(
                    self._p_true_candidates_judge_messages(q, a, candidates),
                    self.model_name
                )
                for q, a, candidates in zip(questions, answers, candidates_list)
            ]

        if self.args.using_host:
            try:
                from tqdm import tqdm
            except ImportError:
                def tqdm(x, *args, **kwargs): return x  # fallback if tqdm is unavailable

            batch_size = 512  # Adjustable batch size, reduce memory pressure
            total = len(prompts)
            for start in tqdm(range(0, total, batch_size), desc="vLLM API batch infer (host)", ncols=80):
                end = min(start + batch_size, total)
                batch_prompts = prompts[start:end]
                batch_results = self.process_batch_llm_host_prompts_log(batch_prompts, temperature=0)
                for text, clean_logprobs_list, tokens in batch_results:
                    p_true, p_false = 0.0, 0.0
                    for token_str, lp in (clean_logprobs_list[0].items() if clean_logprobs_list else []):
                        # Default use logprobs of first token position
                        try:
                            token_lower = token_str.strip().lower()
                        except Exception:
                            token_lower = str(token_str).lower()
                        try:
                            prob = math.exp(lp)
                        except Exception:
                            continue
                        if "true" in token_lower:
                            p_true += prob
                        elif "false" in token_lower:
                            p_false += prob
                    norm = p_true + p_false
                    if norm > 1e-10:
                        p_true = p_true / norm
                        p_false = p_false / norm
                    else:
                        p_true = None
                        p_false = None

                    if candidates_list is None:
                        results.append({
                            "p_true_logits_true": p_true,
                            "p_true_logits_false": p_false,
                            "p_true_logits_conf": p_true,
                            "p_true_logits_raw": text,
                        })
                    else:
                        results.append({
                            "p_true_logits_candidates_true": p_true,
                            "p_true_logits_candidates_false": p_false,
                            "p_true_logits_candidates_conf": p_true,
                            "p_true_logits_candidates_raw": text,
                        })
        else:
            # === Otherwise, use local self.llm.generate ===
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                logprobs=20,
            )

            outputs = self.llm.generate(prompts, sampling_params)
            tokenizer = self.llm.llm_engine.tokenizer.tokenizer

            # --- Internal function ---
            def collect_true_false_variants(logprobs_dict, tokenizer):
                variants = {"true": set(), "false": set()}
                for tok_id in logprobs_dict.keys():
                    tid = tok_id
                    token_str = tokenizer.decode([tid])
                    token_lower = token_str.strip().lower()
                    if "true" in token_lower:
                        variants["true"].add(token_lower)
                    elif "false" in token_lower:
                        variants["false"].add(token_lower)
                return variants

            # === Process local inference results ===
            for out in outputs:
                if not out.outputs or not out.outputs[0].logprobs:
                    results.append({"p_true": None, "p_false": None, "raw": None, "output": None})
                    continue

                logprobs_dict = out.outputs[0].logprobs[0]
                p_true, p_false = 0.0, 0.0
                variants = collect_true_false_variants(logprobs_dict, tokenizer)

                for token_id, lp in logprobs_dict.items():
                    token_str = tokenizer.decode([token_id])
                    token_lower = token_str.strip().lower()

                    if token_lower in variants["true"]:
                        p_true += math.exp(lp.logprob)
                    elif token_lower in variants["false"]:
                        p_false += math.exp(lp.logprob)

                norm = p_true + p_false
                if norm > 1e-10:
                    p_true /= norm
                    p_false /= norm
                else:
                    p_true = None
                    p_false = None

                model_output = out.outputs[0].text.strip() if out.outputs and out.outputs[0].text else None

                if candidates_list is None:
                    results.append({
                        "p_true_logits_true": p_true,
                        "p_true_logits_false": p_false,
                        "p_true_logits_conf": p_true,
                        "p_true_logits_raw": text,
                    })
                else:
                    results.append({
                        "p_true_logits_candidates_true": p_true,
                        "p_true_logits_candidates_false": p_false,
                        "p_true_logits_candidates_conf": p_true,
                        "p_true_logits_candidates_raw": text,
                    })

        return results

    from typing import List, Dict, Optional
    from collections import defaultdict

    def confidence_from_p_true_sampling_batch(
        self,
        questions: List[str],
        answers: List[str],
        candidates_list: Optional[List[Optional[List[str]]]] = None,
        num_samples: int = 10,
        temperature: float = 1,
        top_p: float = 1,
        max_tokens: int = 4
    ) -> List[Dict]:
        """
        Unified version: Automatically select message generation logic based on whether candidates_list is passed.
        Supports using local LLM or host mode (port 8000).
        """
        from typing import List, Dict, Optional
        from collections import defaultdict
        results = []
        all_messages_list = []
        qa_indices = []

        # === Construct messages ===
        for idx, (q, a) in enumerate(zip(questions, answers)):
            candidates = None if candidates_list is None else candidates_list[idx]
            for _ in range(num_samples):
                if candidates_list is None:
                    all_messages_list.append(self._p_true_judge_messages(q, a))
                else:
                    all_messages_list.append(self._p_true_candidates_judge_messages(q, a, candidates))
                qa_indices.append(idx)

        # === Batch Inference ===
        if self.args.using_host:
            outs = self.local_chat_batch_host(
                all_messages_list,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
        elif self.args.using_api:
            outs = self.local_chat_batch_eval_api(all_messages_list)
        else:
            outs = self.local_chat_batch(
                all_messages_list,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )

        # === Parse True/False results ===
        parsed = self._parse_bool_text_batch(outs)

        # === Group statistics ===
        group_results = defaultdict(list)
        for idx, val in zip(qa_indices, parsed):
            group_results[idx].append(val)

        for idx in range(len(questions)):
            group = group_results[idx]
            t = group.count(True)
            f = group.count(False)
            inv = group.count(None)
            denom = max(1, t + f)
            p_true = t / denom
            majority = True if t > f else False if f > t else None
            raw_outs = outs[idx * num_samples : (idx + 1) * num_samples]

            if candidates_list is None:
                results.append({
                    "p_true_sample_conf": p_true,  # Keep two decimal places
                    "p_true_sample_true_num": t,
                    "p_true_sample_false_num": f,
                    "p_true_sample_invalid_num": inv,
                    "p_true_sample_majority": majority,
                    "p_true_sample_raw": raw_outs,
                })
            else:
                results.append({
                    "p_true_sample_candidates_conf": p_true,  # Keep two decimal places
                    "p_true_sample_candidates_true_num": t,
                    "p_true_sample_candidates_false_num": f,
                    "p_true_sample_candidates_invalid_num": inv,
                    "p_true_sample_candidates_majority": majority,
                    "p_true_sample_candidates_raw": raw_outs,
                })

        return results

    # ========= JSONL Batch Processing =========

    def compute_post(
        self,
        num_samples: int = 10,
        use_post_self : bool = False,
        use_post_self_candidates : bool = False,
        use_post_p_true_logits = False,
        use_post_p_true_logits_candidates = False,
        use_post_p_true: bool = False,
        use_post_p_true_candidates : bool = False,
    ):
        
        infile = self.args.outfile
        outfile = self.args.outfile.replace('.jsonl', '_post.jsonl')

        
        # New logic: If output file exists and all key fields in corresponding data are not empty and force overwrite is not set, return directly
        force_replace = self.args.force_replace
        required_fields = [
            "self_report_candidates_conf",
            "self_report_conf",
            "p_true_sample_conf",
            "p_true_sample_candidates_conf",
        ]
        if os.path.exists(outfile) and not force_replace:
            try:
                with open(outfile, "r", encoding="utf-8") as fin:
                    lines = [line for line in fin if line.strip()]
                def is_valid_line(line):
                    try:
                        data = json.loads(line)
                        for f in required_fields:
                            # Field must exist, not be None, and not be empty string, otherwise regarded as invalid
                            if f not in data or data[f] is None or data[f] == "":
                                return False
                        return True
                    except Exception:
                        return False
                has_all = all(is_valid_line(line) for line in lines) and len(lines) > 0
            except Exception as ex:
                has_all = False

            if has_all:
                print(f"Post output exists and all key fields in all lines are non-empty: {outfile}, skipping calculation")
                # Merge into eval_post
                eval_file = self.args.outfile.replace('.jsonl', '_eval.jsonl')
                if os.path.exists(eval_file):
                    with open(eval_file, "r", encoding="utf-8") as fin:
                        eval_lines = [json.loads(line) for line in fin if line.strip()]
                    with open(outfile, "r", encoding="utf-8") as fin:
                        post_lines = [json.loads(line) for line in fin if line.strip()]
                    # Merge: Assuming order of two files is consistent, add has_answer field from eval_lines to each item in post_lines
                    out_eval_post = []
                    for p, e in zip(post_lines, eval_lines):
                        out_p = dict(p)
                        if "has_answer" in e:
                            out_p["has_answer"] = e["has_answer"]
                        out_eval_post.append(json.dumps(out_p, ensure_ascii=False))
                    eval_post_file = self.args.outfile.replace('.jsonl', '_eval_post.jsonl')
                    with open(eval_post_file, "w", encoding="utf-8") as fout:
                        fout.write("\n".join(out_eval_post) + "\n")
                return
                
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        with open(infile, "r", encoding="utf-8") as fin:
            lines = [json.loads(line) for line in fin if line.strip()]

        all_outputs = []

        batch = lines
        qs = [item.get("question", "") for item in batch]
        ans = [item.get("Res", "") for item in batch]
        # Find candidates corresponding to current batch
        confidence_res_file = infile.replace('.jsonl', '_sample.jsonl')
        with open(confidence_res_file, "r", encoding="utf-8") as fin:
            candidates_dict = [json.loads(line) for line in fin if line.strip()]
        # Extract candidate set corresponding to current batch
        candidates_batch = candidates_dict
        candidates_list = [k.get("Res", "") for k in candidates_batch]

        # Self-report confidence
        if use_post_self:
            self_reports_results = self.confidence_from_self_report_batch(qs, ans)
        if use_post_self_candidates:
            self_reports_candidates_results = self.confidence_from_self_report_batch(qs, ans, candidates_list=candidates_list)
        # Sampling confidence
        if use_post_p_true:
            p_true_results = self.confidence_from_p_true_sampling_batch(
                qs, ans, num_samples=num_samples, temperature=1, top_p=1
            )
        if use_post_p_true_candidates:
            p_true_candidates_results = self.confidence_from_p_true_sampling_batch(
                questions=qs, answers=ans, candidates_list=candidates_list, num_samples=num_samples, temperature=1, top_p=1
            )
        # Use open source logits
        if use_post_p_true_logits:
            p_true_logits_results = self.confidence_from_p_true_logits_batch(qs, ans)
        if use_post_p_true_logits_candidates:
            p_true_logits_candidates_results = self.confidence_from_p_true_logits_batch(
                questions=qs, answers=ans, candidates_list=candidates_list
            )

        for j, item in enumerate(batch):
            out = dict(item)
            if use_post_self:
                out.update(self_reports_results[j])
            if use_post_self_candidates:
                out.update(self_reports_candidates_results[j])
            if use_post_p_true:
                out.update(p_true_results[j])
            if use_post_p_true_candidates:
                out.update(p_true_candidates_results[j])
            if use_post_p_true_logits:
                out.update(p_true_logits_results[j])
            if use_post_p_true_logits_candidates:
                out.update(p_true_logits_candidates_results[j])
            out = self.del_log(out)
            all_outputs.append(out)

        with open(outfile, "w", encoding="utf-8") as fout:
            for item in all_outputs:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Output post to {outfile}")

        # Merge into eval_post
        eval_file = self.args.outfile.replace('.jsonl', '_eval.jsonl')
        if os.path.exists(eval_file):
            with open(eval_file, "r", encoding="utf-8") as fin:
                eval_lines = [json.loads(line) for line in fin if line.strip()]
            with open(outfile, "r", encoding="utf-8") as fin:
                post_lines = [json.loads(line) for line in fin if line.strip()]
            # Merge: Assuming order of two files is consistent, add has_answer field from eval_lines to each item in post_lines
            out_eval_post = []
            for p, e in zip(post_lines, eval_lines):
                out_p = dict(p)
                if "has_answer" in e:
                    out_p["has_answer"] = e["has_answer"]
                out_eval_post.append(json.dumps(out_p, ensure_ascii=False))
            eval_post_file = self.args.outfile.replace('.jsonl', '_eval_post.jsonl')
            with open(eval_post_file, "w", encoding="utf-8") as fout:
                fout.write("\n".join(out_eval_post) + "\n")