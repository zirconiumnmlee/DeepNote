import os, time, json, re
import torch
import random
import numpy as np
import yaml
import backoff
import requests

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SILICONFLOW_API_KEY=config["model"]["SILICONFLOW_API_KEY"]
SILICONFLOW_API_URL=config["model"]["SILICONFLOW_API_URL"]

def p_template(template, variables):
    prompt_file = f"../prompts/en/{template}"
    with open(prompt_file, "r") as fin:
        prompt = fin.read()
    prompt = prompt.format(**variables)
    return prompt

def silicon_gen(model, content, temperature=0.1, top_p=0.9):
    try:
        url = SILICONFLOW_API_URL

        payload = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "stream": False,
            "max_tokens": 512,
            "min_p": 0.05,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "stop": [],
            "messages": [{"role": "user","content": content}]
        }

        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        try:
            data = response.json()
        except Exception:
            print("⚠️ 响应不是合法JSON:", response.text)
            raise Exception("API响应格式错误")

        return data["choices"][0]["message"]["content"]

        #return response.text["choices"][0]["message"]["contents"]
    
    except Exception as e:
        print(f"[siliconflow_gen] Error: {e}")
        time.sleep(0.5)
    
    return None

@backoff.on_exception(backoff.expo, (Exception), max_time=300, max_tries=5)
def call_silicon(model, content, temperature=1, top_p=0.9):
    res = silicon_gen(model, content, temperature, top_p)
    if res is None:
        raise Exception("API调用失败")
    return res


def extract_best_worst(sequence):
    try:
        if "json" in sequence:
            sequence = sequence[sequence.index("{") : sequence.rindex("}") + 1]
        result = json.loads(sequence)
        best = result["best_id"]
        worst = result["worst_id"]

    except:
        pattern_best = re.compile(r'"best_id"\s*:\s*(\d+)')
        pattern_worst = re.compile(r'"worst_id"\s*:\s*(\d+)')

        best_match = pattern_best.search(sequence)
        worst_match = pattern_worst.search(sequence)

        best = int(best_match.group(1)) if best_match else None
        worst = int(worst_match.group(1)) if worst_match else None
    if best == worst:
        return None, None
    return best, worst


def LLM_score_rag(responses, answer):
    chosen, rejected = [], []
    for item in responses:
        if item.strip().lower() == answer.lower():
            chosen.append(item)
        if answer.lower() not in item.lower():
            rejected.append(item)
    if len(chosen) == 0:
        for item in responses:
            if answer.strip().lower() in item.lower() and len(item) < 2 * len(answer):
                chosen.append(item)
    if len(rejected) == 0:
        for item in responses:
            if len(item.strip()) > 2 * len(answer):
                rejected.append(item)
    if len(chosen) > 0 and len(rejected) > 0:
        return random.sample(chosen, 1)[0], random.sample(rejected, 1)[0]
    else:
        return "", ""


def LLM_score_init_note(model_name, question, refs, init_notes):
    init_notes_prompt = [
        {"_id": id, "content": content} for id, content in enumerate(init_notes)
    ]
    prompt = config["score"]["init_note"].format(
        query=question, refs=refs, notes=init_notes_prompt
    )
    response = call_silicon(model_name, prompt)
    best, worst = extract_best_worst(response)
    try:
        best_note, worst_note = init_notes[best], init_notes[worst]
    except:
        return init_notes[-1], ""
    return best_note, worst_note


def LLM_score_gen_new_query(
    model_name, question, best_init_note, new_querys, query_log
):
    querys_prompt = [
        {"_id": id, "content": content} for id, content in enumerate(new_querys)
    ]
    prompt = config["score"]["gen_new_query"].format(
        notes=best_init_note,
        query=question,
        query_log=query_log,
        new_querys=querys_prompt,
    )
    response = call_silicon(model_name, prompt)
    best, worst = extract_best_worst(response)

    try:
        best_query, worst_query = new_querys[best], new_querys[worst]
    except:
        return new_querys[-1], ""
    return best_query, worst_query


def LLM_score_compare_note(model_name, query, best_note, new_note):
    prompt = p_template(
        "compare", {"query": query, "best_note": best_note, "new_note": new_note}
    )
    response = call_silicon(model_name, prompt)

    try:
        if "json" in response:
            response = response[response.index("{") : response.rindex("}") + 1]
        status = json.loads(response)["status"]
        return status.lower() == "true"
    except:
        return "true" in response.lower()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)