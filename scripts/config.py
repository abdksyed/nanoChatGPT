from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    
    epochs: int = 3
    lr: float = 1e-4
    accumulation_steps: int = 32 # effective batch size 16*8=128
    label_smoothing: float = 0.1
    batch_size: int = 32
    epochs_save: str = "epoch_{0}_{1:03d}.pth"
    weights_folder: str = "./weights"
    log_dir = "../logs/GPT"

    tokenizer_name: str = "gpt2"

    # device cpu or cuda or mps
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@dataclass
class GPT2Config(GPTConfig):
    vocab_size: int = 50257  # GPT-2 vocab_size of 50257
    block_size: int = 1024
    bias: bool = True
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    model_name: str = "gpt2"
    lora_rank: int = 1


@dataclass
class GPT2medConfig(GPTConfig):
    vocab_size: int = 50257  # GPT-2 vocab_size of 50257
    block_size: int = 1024
    bias: bool = True
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0

    model_name: str = "gpt2-medium"
    lora_rank: int = 16

@dataclass
class GPT2xlConfig(GPTConfig):
    vocab_size: int = 50257  # GPT-2 vocab_size of 50257
    block_size: int = 1024
    bias: bool = True
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600
    dropout: float = 0.0

    model_name: str = "gpt2-xl"
    lora_rank: int = 16


@dataclass
class ScratchConfig:
    tokenizer_file: str = "../data/opus_books/tokenizer_{0}.json"
    src_lang: str = "en"
    tgt_lang: str = "fr"
    batch_size: int = 32
    max_seq_len: int = 512

    # Model
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
