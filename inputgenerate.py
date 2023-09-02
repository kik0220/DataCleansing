import os
import json
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

source_file_path = "input.txt"
save_file_path = "output"

script_dir = os.getcwd()

def source_file_open(source_file):
    with open(source_file, 'r', encoding='utf-8') as file:
        source_text = file.read()
    return source_text.split("\n")

source_texts = source_file_open(source_file_path)

tokenizer = AutoTokenizer.from_pretrained("matsuo-lab/weblab-10b-instruction-sft",cache_dir="models")
model = AutoModelForCausalLM.from_pretrained("matsuo-lab/weblab-10b-instruction-sft",cache_dir="models")
# tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'],cache_dir="models")
# model = AutoModelForCausalLM.from_pretrained("stabilityai/japanese-stablelm-base-alpha-7b",trust_remote_code=True,cache_dir="models")
# tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft-v2", use_fast=False,cache_dir="models")
# model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",cache_dir="models")
# tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b",cache_dir="models")
# model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", device_map="auto", torch_dtype=torch.float16,cache_dir="models")
# tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b",cache_dir="models")
# model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b", torch_dtype="auto",cache_dir="models")
# tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",use_fast=True,cache_dir="models")
# model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",torch_dtype="auto",cache_dir="models")
# tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-1.7b-instruction-sft",legacy=True,cache_dir="models")
# model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-1.7b-instruction-sft",torch_dtype="auto",cache_dir="models")
# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "次の文章は女性の台詞です。32文字程度で要約を出力しなさい。"
model.half()
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")

out_text = ""
for source_text in source_texts:
    prompt = f'{DEFAULT_SYSTEM_PROMPT}\n\n### 女性:\n{source_text}\n\n### 要約:'
    # prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(bos_token=tokenizer.bos_token,b_inst=B_INST,system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",prompt=source_text,e_inst=E_INST,)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    seed = 23  
    torch.manual_seed(seed)

    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=500,
        temperature=0.98,
        top_p=0.37,
        top_k=100,
        typical_p=1,
        repetition_penalty=1.23,
        encoder_repetition_penalty=1,
        no_repeat_ngram_size=0,
        epsilon_cutoff=0,
        eta_cutoff=0,
        min_length=0,
        do_sample=True,
    )

    out = tokenizer.decode(tokens.tolist()[0][input_ids.size(1) :], skip_special_tokens=True)
    out_text += out.replace("\n", "") + "\n"

with open(save_file_path + ".txt", 'w', encoding='utf-8') as file:
    file.write(out_text)

out_text = []
out_texts = source_file_open(save_file_path + ".txt")
for i in range(0, len(source_texts), 1):
    out_text += {"instruction": out_texts[i], "output": source_texts[i]},

with open(save_file_path + ".json", 'w', encoding='utf-8') as file:
    json.dump(out_text, file, ensure_ascii=False, indent=4)
