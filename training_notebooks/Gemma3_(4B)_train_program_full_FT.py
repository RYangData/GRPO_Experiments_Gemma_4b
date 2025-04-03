# %% [markdown]
# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# %% [markdown]
# ### News

# %% [markdown]
# **Read our [Gemma 3 blog](https://unsloth.ai/blog/gemma3) for what's new in Unsloth and our [Reasoning blog](https://unsloth.ai/blog/r1-reasoning) on how to train reasoning models.**
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# %% [markdown]
# ### Installation

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
    !pip install --no-deps unsloth vllm
# Install latest Hugging Face for Gemma-3!
!pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# %%
#@title Colab Extra Install { display-mode: "form" }
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    !pip install --no-deps unsloth vllm
    # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
    # Skip restarting message in Colab
    import sys, re, requests; modules = list(sys.modules.keys())
    for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer

    # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
    f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
    with open("vllm_requirements.txt", "wb") as file:
        file.write(re.sub(rb"(transformers|numpy|xformers)[^\n]{1,}\n", b"", f))
    !pip install -r vllm_requirements.txt

# %%
!pip install bitsandbytes

# %%
!pip install unsloth_zoo

# %%
!pip install msgspec

# %%
!pip install blake3

# %%
!pip install gguf

# %%
# Standard imports
import os
import torch
from unsloth import FastModel
from datasets import Dataset

import json
from pathlib import Path

# 1. Load your processed datasets
base_dir = "/content/drive/MyDrive/2025_ConvFinQA_SFT_Agentic"
data_dir = Path(f"{base_dir}/data/processed_datasets")

# Choose which format to use (Objective A or B)
# Objective A: Final answer only
# Objective B: Program + Answer (recommended for better reasoning)
training_format = "program_answer"  # Change to "final_answer" if needed

# Load train and evaluation data
train_path = data_dir / f"finqa_{training_format}_train.jsonl"
dev_path = data_dir / f"finqa_{training_format}_dev.jsonl"

# Load data as lists
with open(train_path, 'r') as f:
    train_data = [json.loads(line) for line in f]

with open(dev_path, 'r') as f:
    dev_data = [json.loads(line) for line in f]

print(f"Loaded {len(train_data)} training examples and {len(dev_data)} development examples")

# 2. Format data according to Gemma 3 chat template
# Convert to HF datasets format
def create_dataset(data, objective="program_answer"):
    if objective == "program_answer":
        # For program + answer format, we need prompt and output
        formatted_data = []
        for example in data:
            formatted_data.append({
                "conversations": [
                    {"role": "user", "content": example["prompt"].replace("### Response:\n", "")},
                    {"role": "assistant", "content": example["output"]}
                ]
            })
    else:
        # For final answer only format
        formatted_data = []
        for example in data:
            formatted_data.append({
                "conversations": [
                    {"role": "user", "content": example["prompt"].replace("### Response:\n", "")},
                    {"role": "assistant", "content": example["answer"]}
                ]
            })

    return Dataset.from_list(formatted_data)

# Create datasets
train_dataset = create_dataset(train_data, objective=training_format)
eval_dataset = create_dataset(dev_data, objective=training_format)

print(f"Prepared {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")

# 3. Initialize Gemma 3 model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",  # 4-bit quantized model
    max_seq_length = 4096,  # Financial texts can be long, set appropriate length
    dtype = None,  # Keep in 4-bit
    load_in_4bit = True,
    # full_finetuning = True
)

# # 4. Add LoRA adapters for efficient fine-tuning
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 16,             # Higher rank for complex financial reasoning
    lora_alpha = 32,    # Alpha typically 2x rank
    lora_dropout = 0.05,  # Small dropout to prevent overfitting
    bias = "none",
    random_state = 2,
)


# %%

# 5. Set up tokenizer with Gemma 3 chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# 6. Apply chat template to datasets
def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}

train_dataset = train_dataset.map(apply_chat_template, batched=True)
eval_dataset = eval_dataset.map(apply_chat_template, batched=True)


# %%

# 7. Set up trainer
from trl import SFTTrainer, SFTConfig

# training_arguments = SFTConfig(
#     output_dir = f"{base_dir}/finqa-gemma3-{training_format}",
#     dataset_text_field = "text",
#     per_device_train_batch_size = 2,
#     gradient_accumulation_steps = 4,
#     warmup_ratio = 0.03,
#     num_train_epochs = 3,       # Adjust based on dataset size
#     learning_rate = 2e-4,       # Higher LR for financial domain
#     max_grad_norm = 0.3,        # Lower to prevent overfitting
#     logging_steps = 10,
#     evaluation_strategy = "steps",
#     eval_steps = 100,           # Evaluate regularly
#     save_strategy = "steps",
#     save_steps = 100,
#     optim = "adamw_8bit",
#     weight_decay = 0.01,
#     lr_scheduler_type = "cosine",
#     seed = 3407,
#     report_to = "none",         # Can use "wandb" if you want to track metrics
# )

training_arguments = SFTConfig(
    output_dir=f"{base_dir}/finqa-gemma3-{training_format}",
    dataset_text_field="text",
    per_device_train_batch_size=2,   # larger batch if GPU allows
    gradient_accumulation_steps=4,
    num_train_epochs=1,              # minimal epochs
    # max_steps=50,                    # explicitly limit steps
    learning_rate=2e-4,
    logging_steps=10,
    # evaluation_strategy="no",        # disables evaluations during training
    # save_strategy="no",              # disables saving checkpoints
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=0,
    report_to="none",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    # eval_dataset = eval_dataset,
    args = training_arguments,
)

# 8. Train only on assistant responses (improves accuracy)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# 9. Verify that masking works properly
print("Example of masked training data:")
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " ")

# 10. Start training
print("Starting training...")
trainer_stats = trainer.train()
print("Training complete!")

# 11. Save the model
output_dir = f"{base_dir}/finqa-gemma3-{training_format}-full"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# 12. Test the trained model
test_examples = dev_data[:3]  # Test on a few examples from dev set

print("\nTesting model on examples:")
for i, example in enumerate(test_examples):
    # Create message format
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": example["prompt"].replace("### Response:\n", "")
        }]
    }]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )

    # Generate response
    outputs = model.generate(
        **tokenizer([text], return_tensors="pt").to("cuda"),
        max_new_tokens = 512,
        temperature = 0.1,  # Lower temperature for financial reasoning
        top_p = 0.95,
        top_k = 50,
    )

    # Print results
    print(f"\nExample {i+1}:")
    print("Input:", example["prompt"].replace("### Response:\n", "")[:200] + "...")
    print("\nExpected output:", example.get("output", example.get("answer", ""))[:200] + "...")
    print("\nModel output:", tokenizer.decode(outputs[0]).split("<start_of_turn>model\n")[1][:200] + "...")
    print("-" * 80)

# %%



