# Based on https://github.com/karpathy/nanoGPT/blob/master/model.py

import math

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import TiktokenTokenizer

import peft
from peft import LoraConfig

from config import GPT2medConfig

# Seeding for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        assert cfg.n_embd % cfg.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # output projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # regularization
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        # Use Flash Attention if available
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                    1, 1, cfg.block_size, cfg.block_size
                ),
            )

    def forward(self, x, attention_mask=None):
        batch, seq_len, emb_dim = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Using View and Transpose
        # k = k.view(batch, seq_len, self.n_head, emb_dim // self.n_head).transpose(1, 2)

        # (batch, seq_len, emb_dim) -> (batch, n_heads, seq_len, head_dim)
        k = einops.rearrange(k, "b s (h d) -> b h s d", h=self.n_head)
        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.n_head)
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.n_head)

        # causal self-attention; Self-attend:
        # (batch, nnum_heads, seq_len, head_size) x (batch, num_heads, head_size, seq_len) -> (batch, num_heads, seq_len, seq_len)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None, # Both attention_mask and is_causal can't be set.
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.tril[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
            if attention_mask is not None:
                # mask out attention for padding tokens
                # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
                # eg: tensor([[1,1,1,0,0], [1,1,0,0,0]]) -> tensor([ [[[1., 1., 1., 0., 0.]]], [[[1., 1., 0., 0., 0.]]] ])
                att = att.masked_fill(attention_mask[:, None, None, :], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = (
                att @ v
            )  # (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, head_size) -> (batch, num_heads, seq_len, head_size)

        y = (
            y.transpose(1, 2).contiguous().view(batch, seq_len, emb_dim)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg, print_params=True) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = TiktokenTokenizer("gpt2")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias),
            )
        )

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def get_num_params(self, trainable=True):
        
        # report number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (total_params / 1e6,))

        token_emb_params = sum(p.numel() for p in self.transformer.wte.parameters())
        pos_emb_params = sum(p.numel() for p in self.transformer.wpe.parameters())
        # Embedding Parameters
        print(
            "Token Embedding Parameters:", f"{token_emb_params/1e6:.2f}", "M parameters"
        )
        print(
            "Position Embedding Parameters:",
            f"{pos_emb_params/1e6:.2f}",
            "M parameters",
        )

        # Model without Embedding Parameters
        transformer_params = total_params - token_emb_params - pos_emb_params
        transformer_params += sum(p.numel() for p in self.transformer.ln_f.parameters())
        print(
            "Transformer Parameters:",
            f"{transformer_params/1e6:.2f}",
            "M parameters",
        )
        # if trainable:
        #     return sum(p.numel() for p in self.parameters() if p.requires_grad)
        # else:
        #     return sum(p.numel() for p in self.parameters())

    def forward(self, idx, attention_mask=None, labels=None):
        bs, t = idx.size()
        assert (
            t <= self.cfg.block_size
        ), f"Cannot forward sequence of length {t} > block_size {self.cfg.block_size}"
        # Embeddings
        tok_emb = self.transformer.wte(idx)
        # print("Token Emb:", tok_emb.shape)
        pos_emb = self.transformer.wpe(torch.arange(t, device=idx.device))
        # print("Pos Emb:", pos_emb.shape)
        x = tok_emb + pos_emb
        # print("x:", x.shape)
        # Transformer
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        # print("After Transformer Blocks:", x.shape)
        x = self.transformer.ln_f(x)
        # print("After Projection Layer:", x.shape)
        # Language Model Head
        if labels is not None:
            logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(
                x[:, -1:, :]
            )  # take only the last token for inference
            loss = None
        # print("Logits:", logits.shape)

        return logits, loss

    def from_pretrained(self, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)
        # # n_layer, n_head and n_embd are determined from model_type
        # config_args = {
        #     "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        #     "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        #     "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        #     "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        # }[model_type]
        # print("forcing vocab_size=50257, block_size=1024, bias=True")
        # config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        # config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # config_args["bias"] = True  # always True for GPT model checkpoints

        # cfg = GPTConfig(**config_args)
        # model = GPT(cfg)
        sd = self.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return self
    
    def from_finetuned(self, path):
        self.load_state_dict(torch.load(path)["model_state_dict"], strict=True)
        return self

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cropped)  # (batch, 1, vocab_size)
            # Apply temperature
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
            if top_k is not None:
                # Select top_k tokens from logits and make all else -inf
                # When softmax is applied only top_k tokens are considered.
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            # apply softmax
            probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            # append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, curr_seq_len+1)

            # Break if end token is generated
            if idx_next.squeeze().item() == self.tokenizer.encode("<|endoftext|>")[0]:
                break

        return idx


if __name__ == "__main__":
    # cfg = GPT2medConfig()
    # model = GPT(cfg).from_pretrained(cfg.model_name)
    # prompt = """There are 100 awesome pokemons, listing them all as follows:
    # Pikachu,
    # Charizard,
    # Bulbasaur,
    # Mew,
    # Genghis,
    # Mewto,
    # Squirtle,
    # Astroz,
    # """
    # inp = model.tokenizer(prompt, return_tensors="pt")["input_ids"].unsqueeze(0)
    # print("Input:", inp.shape)
    # out = model.generate(inp, 200, temperature=1.0, top_k=3)
    # print("Output:", out.shape)
    # print(model.tokenizer.decode(out[0].tolist()))

    cfg = GPT2medConfig()
    model = GPT(cfg).from_pretrained(cfg.model_name)
    target_modules = []
    for n, m in model.named_modules():
        if "c_attn" in n or "c_proj" in n or "c_fc" in n:
            target_modules.append(n)
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        target_modules=target_modules,
        modules_to_save=["transformer.wte", "transformer.wpe", "lm_head"],
    )
    # Print lora trainable parameters
    peft_model = peft.get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
