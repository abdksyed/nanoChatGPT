import torch
import torch.nn as nn
import einops

from gpt import GPT


class KPairWiseLoss(nn.Module):
    # TODO: Check the logic of this loss

    def forward(self, scores):
        # scores in each of batch, should be ranked in descending order
        B, N = scores.shape  # Ex: (2,3)

        scores = scores.view(B, N, 1)  # (B,N,1)

        # To get difference between all pairs in vectorized form
        a = scores.tile((1, N))  # (B,N,N)
        b = a.transpose(1, 2)  # (B,N,N)

        log_pair_diff = nn.functional.logsigmoid((b - a))  # (B,N,N)

        # tril to remove duplication in pairs and also diagonals to remove self comparison
        tril = torch.tril(torch.ones(B, N, N), diagonal=-1)
        # log_pair_diff * tril will remove all the pairs which are not in lower triangle
        nll = -1 * (log_pair_diff * tril).sum(dim=(1, 2))

        # total number of comparisons for each score in batch
        total_comparison = tril.sum(dim=(1, 2))

        loss = nll / total_comparison

        return loss


class PairWiseLoss(nn.Module):
    def forward(self, scores, response_mask):
        loss = nn.functional.logsigmoid(scores) * response_mask
        loss = -1 * (loss.sum() / response_mask.sum())

        return loss


class GPTReward(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = GPT(cfg)
        # removing the last linear layer
        self.backbone.lm_head = nn.Identity()
        # Adding new head, with 1 output
        self.lm_head = nn.Linear(cfg.n_embd, 1, bias=False)

    def get_num_params(self, trainable=True):
        # report number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (total_params / 1e6,))

        token_emb_params = sum(
            p.numel() for p in self.backbone.transformer.wte.parameters()
        )
        pos_emb_params = sum(
            p.numel() for p in self.backbone.transformer.wpe.parameters()
        )
        # Embedding Parameters
        print(
            "Token Embedding Parameters:", f"{token_emb_params/1e6:.2f}", "M parameters"
        )
        print(
            "Position Embedding Parameters:",
            f"{pos_emb_params/1e6:.2f}",
            "M parameters",
        )
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        print(
            "Final Linear Head Parameters:",
            f"{lm_head_params/1e6:.4f}",
            "M parameters",
        )

        # Model without Embedding Parameters
        transformer_params = (
            total_params - token_emb_params - pos_emb_params - lm_head_params
        )
        print(
            "Transformer Parameters:",
            f"{transformer_params/1e6:.2f}",
            "M parameters",
        )

    def forward(self, x, attention_mask, loss: int = 2):
        x = einops.rearrange(x, "b k n -> (b k) n")  # (B,2,T) -> (B*2,T)
        attention_mask, response_mask = attention_mask
        attention_mask = einops.rearrange(attention_mask, "b k n -> (b k) n")
        x, _ = self.backbone(x, attention_mask)  # (B*2,T,D(768))
        x = self.lm_head(x).squeeze(-1)  # (B*2,T)

        if loss is not None:
            # loss: 2 if there are only 2 responses (pos and neg)
            # output contains [pos, neg, pos, neg, ...]
            # To make score as [[pos, neg], [pos, neg], ...]
            x = einops.rearrange(x, "(b n) t -> b n t", n=loss)
            pos, neg = x[:, 0, :], x[:, 1, :]
            scores = pos - neg

            return x, PairWiseLoss()(scores, response_mask)

        return x, None

    def from_pretrained(self, model_name=None, path=None):
        if model_name is not None:
            self.backbone = GPT(self.cfg).from_pretrained(model_name)
            self.backbone.lm_head = nn.Identity()
            self.lm_head = nn.Linear(self.cfg.n_embd, 1, bias=False)
            return self
        elif path is not None:
            self.backbone = GPT(self.cfg).from_finetuned(path)
            self.backbone.lm_head = nn.Identity()
            self.lm_head = nn.Linear(self.cfg.n_embd, 1, bias=False)
            return self
        else:
            raise ValueError("model_name or path should be provided")
