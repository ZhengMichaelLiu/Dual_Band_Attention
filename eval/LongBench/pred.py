import os
from datasets import load_dataset
import torch
import json
from transformers import GenerationConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from DBA.src.load_llama import load_model, FastPrefillConfig

# models="llama3.1-8b-instruct"
# tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
# method in full minfer flex xattn dba
# stride in 8 16

# python -u eval/LongBench/pred.py --model $model --task $task --method $method
# python -u eval/LongBench/pred.py --model $model --task $task --method $method --stride $stride

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.1-8b-instruct", choices=["llama3.1-8b-instruct",])
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--method", type=str, default="dba", choices=["full", "minfer", "flex", "xattn", "dba"])
    parser.add_argument("--stride", type=int, default=8, help="stride for xattn or dba")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama3" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "llama3" in model_name:
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    return response

def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
):
    preds = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        pbar.set_description(f"Generating for {idx}, len = {input.input_ids.shape[-1]}")
        with torch.no_grad():
            output = model(
                input_ids=input.input_ids,
                past_key_values=None,
                use_cache=True,
                num_logits_to_keep=1,
            )
            past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                    num_logits_to_keep=1,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        print(f"Prediction: {pred}")
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    with open("eval/LongBench/config/model2path.json", "r") as f:
        model2path = json.load(f)
    for k, v in model2path.items():
        if not os.path.isabs(v):
            model2path[k] = os.path.join(project_root, v)
    with open("eval/LongBench/config/model2maxlen.json", "r") as f:
        model2maxlen = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model

    if args.method in ["xattn", "dba"]:
        fast_prefill_config = FastPrefillConfig(stride=args.stride, metric=args.method)
    elif args.method in ["full", "minfer", "flex"]:
        fast_prefill_config = FastPrefillConfig(metric=args.method)
    model, tokenizer = load_model(fast_prefill_config, model2path[model_name])
    generation_config = GenerationConfig.from_pretrained(model2path[model_name])
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    max_length = model2maxlen[model_name]
    datasets = [args.task]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    with open("eval/LongBench/config/dataset2prompt.json", "r") as f:
        dataset2prompt = json.load(f)
    with open("eval/LongBench/config/dataset2maxlen.json", "r") as f:
        dataset2maxlen = json.load(f)

    # predict on each dataset
    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")
    for dataset in datasets:
        data = load_dataset("THUDM/LongBench", dataset, split="test", trust_remote_code=True)
        if not os.path.exists(f"eval/LongBench/pred/{model_name}"):
            os.makedirs(f"eval/LongBench/pred/{model_name}")
        if args.method == "full":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-full.jsonl"
        elif args.method == "minfer":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-minfer.jsonl"
        elif args.method == "flex":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-flex.jsonl"
        elif args.method == "xattn":
            if args.stride == 8:
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-xattn-stride=8.jsonl"
            else:
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-xattn-stride=16.jsonl"
        elif args.method == "dba":
            if args.stride == 8:
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-dba-stride=8.jsonl"
            else:
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-dba-stride=16.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
