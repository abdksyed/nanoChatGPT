import torch
import tiktoken


class TiktokenTokenizer:
    def __init__(self, name) -> None:
        self.enc = tiktoken.get_encoding(name)
        # the `<|endoftext|>` is present in disallowed_special
        # hence we need to add to allow_special tokens to encode it.
        self.encode = lambda s: self.enc.encode(s, allowed_special={"<|endoftext|>"})
        self.pad_token = self.enc.eot_token  # <|endoftext|>

    def __call__(
        self, text, max_length=None, padding=None, truncation=False, return_tensors=None
    ):
        ids = self.encode(text)  # get ids of text ([int, ..., int])
        if truncation:  # if truncation is True
            ids = ids[:max_length]
        # mask for actual tokens present with value `1`
        mask = [1] * len(ids)
        if padding == "max_length":
            # attach [0] to mask depicting padding
            mask += [0] * (max_length - len(ids))
            # attach self.pad_token to ids depicting padding
            ids += [self.pad_token] * (max_length - len(ids))

        if return_tensors == "pt":
            # convert to torch tensors
            ids = torch.tensor(ids, dtype=torch.long)
            mask = torch.tensor(mask)

        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids):
        return self.enc.decode(ids)
