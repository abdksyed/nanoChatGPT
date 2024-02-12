import os

from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk

from peft import LoraConfig, get_peft_model

import wandb

from config import GPT2medConfig
from gpt import GPT
from gpt_rm import GPTReward
from trainer import train
from tokenizer import TiktokenTokenizer
from dataset import DahaosRLHF


cfg = GPT2medConfig()
cfg.batch_size = 8
cfg.accumulation_steps = 16  # effective batch: 4*32=128
cfg.epochs = 10
if cfg.device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

tokenizer = TiktokenTokenizer(cfg.tokenizer_name)

# dataset = load_dataset("stingning/ultrachat") # Dahoas/full-hh-rlhf
# sft_data = dataset["train"].train_test_split(test_size=0.2)
# train_data, val_data = sft_data["train"], sft_data["test"]

dataset = load_dataset("Dahoas/full-hh-rlhf")
sft_data = dataset["train"].train_test_split(test_size=0.2)
train_data, val_data = sft_data["train"], sft_data["test"]

train_data = DahaosRLHF(train_data, tokenizer, block_size=1024)
val_data = DahaosRLHF(val_data, tokenizer, block_size=1024)

# train_data = load_from_disk("./data/ultrachat/sft_train")
# val_data = load_from_disk("./data/ultrachat/sft_val")

print("Number of rows in train data:", len(train_data))
print("Number of rows in validation data:", len(val_data))

cfg.train_data = len(train_data)
cfg.val_data = len(val_data)

wandb.init(project="nanoChatGPT", name="gpt2med-reward", config=cfg)

train_loader = DataLoader(
    train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True
)
val_loader = DataLoader(
    val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True
)

model = GPTReward(cfg).from_pretrained(path="../weights/sft_gpt2-medium.pth")


# LoRA
target_modules = []
for n, m in model.named_modules():
    if "c_attn" in n or "c_proj" in n or "c_fc" in n:
        target_modules.append(n)
lora_cfg = LoraConfig(
    r = cfg.lora_rank,
    target_modules=target_modules,
    # modules_to_save=["transformer.wte", "transformer.wpe", "lm_head"],
)

# Print lora trainable parameters
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

train(cfg, model, train_loader, val_loader)
wandb.finish()
