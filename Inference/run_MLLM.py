import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
from utils.llm import ApiGenerator
from utils.data import QADataset, QADataset_ptb, VQADataset_disc
from utils.llm_local import LocalGenerator
import os
import argparse

ra_dict = {
    'none': 'none',
    'sparse': {'sparse_ctxs': 1},
    'dense': {'dense_ctxs': 1},
    'chatgpt': {'gen_ctxs': 100},
    'sparse+dense': {'dense_ctxs': 5, 'sparse_ctxs': 5},
    'gold': {'gold_ctxs': 1},
    'strong': {'strong_ctxs': 10},
    'weak': {'weak_ctxs': 10},
    'rand': {'rand_ctxs': 10},
    'dpr': {'dpr_ctx': 1},
    'extract': {'dpr_ctx': 1},
    'dpr_wrong': {'dpr_ctx_wrong': 1}
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/source/')  # now is a directory
    parser.add_argument('--relative_prefix', type=str, default='')
    parser.add_argument('--response', type=str, default='')
    parser.add_argument('--usechat', action='store_true')
    parser.add_argument('--using_post', action='store_true')
    parser.add_argument('--using_multi_answer_sme', action='store_true')
    parser.add_argument('--using_multi_answer_prob_add', action='store_true')
    parser.add_argument('--using_multi_answer_plogp', action='store_true')
    parser.add_argument('--consistency_origin', action='store_true')
    parser.add_argument('--using_host', action='store_true')
    parser.add_argument('--using_api', action='store_true')
    parser.add_argument('--prob_add_thres', type=float)
    parser.add_argument('--local_image', type=bool, default=False)
    parser.add_argument('--type', type=str, default='vqa')
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outdir', type=str, default='data/qa')
    parser.add_argument('--outfile', type=str, default='data/qa/.json')
    parser.add_argument('--idx', type=str, default="")
    parser.add_argument('--confidence_dir', type=str, default="")
    parser.add_argument('--greedy_dir', type=str, default="")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--task', type=str, default='nq')
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--hidden_states', type=bool, default=False)
    parser.add_argument('--output_states', type=bool, default=False)
    parser.add_argument('--attn_weights', type=bool, default=False)
    parser.add_argument('--hidden_idx_mode', type=str, default='last')
    parser.add_argument('--need_layers', type=str, default='last', choices=['all', 'last', 'mid'])
    parser.add_argument('--use_api', type=bool, default=False) # whether to use a model api
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--answer_match_model', type=str, default='') # whether to use a model to match answer,if not, using answer containing.
    parser.add_argument('--answer_match_model_api', type=bool, default=False)
    parser.add_argument('--eval_model_name', type=str, default='')
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--start_line', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='llm')
    parser.add_argument('--using_consistency', action="store_true")
    parser.add_argument('--using_sample', action="store_true")
    parser.add_argument('--using_output_all', action="store_true")
    parser.add_argument('--using_vanilla_verb', action="store_true")
    parser.add_argument('--using_sme', action="store_true")
    parser.add_argument('--using_topk_verb', action="store_true")
    parser.add_argument('--consistency_origin_weight_vanilla', action="store_true")
    parser.add_argument('--consistency_origin_weight_topk', action="store_true")
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--using_latent', action="store_true")
    parser.add_argument('--consistency_perturb', type=bool, default=False)
    parser.add_argument('--consistency_num', type=int, default=1)
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--num_q', type=int, default=10)
    parser.add_argument('--answer_judge', type=str, default='in_answer')
    parser.add_argument('--describe_img', type=bool, default=False)
    parser.add_argument('--description_path', type=str, default='')
    parser.add_argument('--image_noise', type=bool, default=False)
    parser.add_argument('--image_noise_start', type=int, default=0)
    parser.add_argument('--image_noise_step', type=int, default=22)
    parser.add_argument('--answer_match_model_bsz', type=int, default=512)
    parser.add_argument('--multi_step_type', type=str, default=None)
    parser.add_argument('--data_num', type=str, default=None)
    parser.add_argument('--max_concurrent', type=int, default=10)
    parser.add_argument('--stream_output', type=bool, default=False)
    parser.add_argument('--logprobs', type=bool, default=False)
    # entity: list of entities to process; if not specified, will automatically traverse all jsonl files in the directory
    parser.add_argument('--entity', type=str, default=None, help='List of entities to process; if not specified, will automatically traverse all jsonl files in the directory')
    parser.add_argument('--force_replace', action="store_true", default=False)

    args = parser.parse_args()
    args.ra = ra_dict[args.ra]
    if not args.model_name:
        args.model_name = args.model_path.split('/')[-1].replace('_', '-').lower()
    return args



def main():
    args = get_args()
    source_dir = args.source
    data_num = args.data_num
    # Automatically get entity list
    if args.entity != "all_entities":
        entities = [args.entity]
    else:
        entities = ["award", "region", "language", "math", "office", "river"]

    # Number of answers to enumerate
    answer_nums = [1, 2, 4, 6]

    if args.use_api:
        engine = ApiGenerator(args)
    else:
        engine = LocalGenerator(args)
    for entity in entities:
        engine.args.entity = entity
        for ans_num in answer_nums:
            # Construct input file name
            input_file = os.path.join(
                source_dir, entity, f"qa_{entity}_{ans_num}_{ans_num}_{data_num}.jsonl"
            )
            if not os.path.exists(input_file):
                print(f"Input file for entity {entity} not found: {input_file}, skipping.")
                continue

            # Automatically generate output file name
            out_dir = os.path.join(args.outdir, args.model_name, entity)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{entity}_{ans_num}a_{data_num}.jsonl")

            print(f"Processing entity: {entity}, number of answers: {ans_num}")
            print(f"Input file: {input_file}")
            print(f"Output file: {out_file}")

            args.source = input_file
            args.outfile = out_file

            if args.using_vanilla_verb:
                if args.using_output_all:
                    args.type = "qa_short_vanilla_verb_output_all"
                else:
                    args.type = "qa_short_vanilla_verb"
            elif args.using_topk_verb:
                if args.using_output_all:
                    args.type = "qa_short_topk_verb_output_all"
                else:
                    args.type = "qa_short_topk_verb"
            else:
                if args.using_output_all:
                    args.type = "qa_short_output_all"
                else:
                    args.type = "qa_short"

            print(args.type)

            max_len = None
            if args.model_type == 'llm':
                all_data = QADataset(args, max_len)

            engine.load_data(all_data)
            engine.get_res()

if __name__ == '__main__':
    main()
