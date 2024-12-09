#nishijimat/llm-jp-3-13b-it

!pip uninstall unsloth -y
!pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Google Colab のデフォルトで入っているパッケージをアップグレード（Moriyasu さんありがとうございます）
!pip install --upgrade torch
!pip install --upgrade xformers

# notebookでインタラクティブな表示を可能とする（ただし、うまく動かない場合あり）
!pip install ipywidgets --upgrade

# Install Flash Attention 2 for softcapping support
import torch
if torch.cuda.get_device_capability()[0] >= 8:
    !pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"

# llm-jp/llm-jp-3-13bを4bit量子化のqLoRA設定でロード。

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel
import torch
max_seq_length = 512 # unslothではRoPEをサポートしているのでコンテキスト長は自由に設定可能
dtype = None # Noneにしておけば自動で設定
load_in_4bit = True # 今回は8Bクラスのモデルを扱うためTrue

model_id = "llm-jp-3-13b"
new_model_id = "llm-jp-3-13b-it" #Fine-Tuningしたモデルにつけたい名前、it: Instruction Tuning
# FastLanguageModel インスタンスを作成
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
)

# SFT用のモデルを用意
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    max_seq_length = max_seq_length,
)

from google.colab import output
output.disable_custom_widget_manager()

# Hugging Face Token を指定
HF_TOKEN = "" #@param {type:"string"}

!pip install datasets

zip_file_path = '/content/drive/MyDrive/Distribution20241221_all.zip'
# 解凍先のディレクトリを指定（任意）
extract_dir = 'extracted_files'

# zip ファイルを解凍
!unzip -q {zip_file_path} -d {extract_dir}

# 学習に用いるデータセットの指定
# https://liat-aip.sakura.ne.jp/wp/llmのための日本語インストラクションデータ作成/llmのための日本語インストラクションデータ-公開/

from datasets import load_dataset

dataset = load_dataset("json", data_files="/content/extracted_files/Distribution20241221_all/ichikara-instruction-003-001-1.json")

# 学習時のプロンプトフォーマットの定義
prompt = """### 指示
{}
### 回答
{}"""



"""
formatting_prompts_func: 各データをプロンプトに合わせた形式に合わせる
"""
EOS_TOKEN = tokenizer.eos_token # トークナイザーのEOSトークン（文末トークン）
def formatting_prompts_func(examples):
    input = examples["text"] # 入力データ
    output = examples["output"] # 出力データ
    text = prompt.format(input, output) + EOS_TOKEN # プロンプトの作成
    return { "formatted_text" : text, } # 新しいフィールド "formatted_text" を返す
pass

# # 各データにフォーマットを適用
dataset = dataset.map(
    formatting_prompts_func,
    num_proc= 4, # 並列処理数を指定
)

dataset

# データを確認
print(dataset["train"]["formatted_text"][3])


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset["train"],
    max_seq_length = max_seq_length,
    dataset_text_field="formatted_text",
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        logging_steps = 10,
        warmup_steps = 10,
        save_steps=100,
        save_total_limit=2,
        max_steps=-1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        group_by_length=True,
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)


#@title 現在のメモリ使用量を表示
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#@title 学習実行
trainer_stats = trainer.train()
# データセットの読み込み。
import json
datasets = []
with open("/content/drive/MyDrive/elyza-tasks-100-TV_0.jsonl", "r") as f:
    item = ""
    for line in f:
      line = line.strip()
      item += line
      if item.endswith("}"):
        datasets.append(json.loads(item))
        item = ""

# 学習したモデルを用いてタスクを実行
from tqdm import tqdm

# 推論するためにモデルのモードを変更
FastLanguageModel.for_inference(model)

results = []
for dt in tqdm(datasets):
  input = dt["input"]

  prompt = f"""### 指示\n{input}\n### 回答\n"""

  inputs = tokenizer([prompt], return_tensors = "pt").to(model.device)

  outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True, do_sample=False, repetition_penalty=1.2)
  prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n### 回答')[-1]

  results.append({"task_id": dt["task_id"], "input": input, "output": prediction})

import os

with open(f"/content/drive/MyDrive/{new_model_id}_output.jsonl", 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')
# jsonlで保存
with open(f"/content/drive/MyDrive/{new_model_id}_output.jsonl", 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

# モデルとトークナイザーをHugging Faceにアップロード。
model.push_to_hub_merged(
    new_model_id,
    tokenizer=tokenizer,
    save_method="lora",
    token=HF_TOKEN,
    private=True
)

