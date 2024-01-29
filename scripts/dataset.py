import random
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = data


class UltraChat(BaseDataset):
    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):
        text_list = self.data[index]["data"]
        final_text = ""
        for idx, q in enumerate(text_list):
            if idx % 2 == 0:
                final_text += "\nHuman: " + q
            else:
                final_text += "\nAsistant: " + q
        final_text = self.tokenizer(final_text)["input_ids"]
        if len(final_text) <= self.block_size + 2:  #
            final_text += [self.tokenizer.pad_token] * (
                self.block_size - len(final_text) + 2
            )
            # print(len(final_text[:self.block_size]), len(final_text[1:1+self.block_size]))
            return (
                torch.tensor(final_text[: self.block_size], dtype=torch.long),
                torch.tensor(final_text[1 : self.block_size + 1], dtype=torch.long),
            )
        else:
            rand_int = random.randint(
                0, len(final_text) - self.block_size - 2
            )  # -2 since we need atleast last token as response
            query = final_text[rand_int : rand_int + self.block_size]
            response = final_text[rand_int + 1 : rand_int + 1 + self.block_size]
            # print(len(query), len(response))
            return torch.tensor(query, dtype=torch.long), torch.tensor(
                response, dtype=torch.long
            )


class DahaosRLHF(BaseDataset):
    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):
        text_dict = self.data[index]
        prompt = text_dict["prompt"]
        prompt_token = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = len(prompt_token["input_ids"])
        # If prompt itself is greater than block_size
        if prompt_len >= self.block_size:
            inputs = prompt_token["input_ids"][: self.block_size]
            inputs = torch.stack((inputs, inputs), dim=0)
            attention_mask = prompt_token["attention_mask"][: self.block_size]
            attention_mask = torch.stack((attention_mask, attention_mask), dim=0)
            response = torch.tensor([0] * self.block_size)
            return inputs, (attention_mask, response)

        pos_text = prompt + text_dict["chosen"]
        pos_text = self.tokenizer(
            pos_text,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg_text = prompt + text_dict["rejected"]
        neg_text = self.tokenizer(
            neg_text,
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = torch.stack((pos_text["input_ids"], neg_text["input_ids"]), dim=0)
        attention_mask = torch.stack(
            (pos_text["attention_mask"], neg_text["attention_mask"]), dim=0
        )

        response_mask = torch.tensor(
            [0] * prompt_len + [1] * (self.block_size - prompt_len)
        )
        response_mask = torch.stack((response_mask, response_mask), dim=0)
        response_mask = attention_mask & response_mask
        response_mask = (
            response_mask[0]
            if response_mask[0].sum() > response_mask[1].sum()
            else response_mask[1]
        )

        return input_ids, (attention_mask, response_mask)


if __name__ == "__main__":
    from tokenizer import TiktokenTokenizer
    from datasets import load_from_disk

    val_data = load_from_disk("./data/ultrachat/sft_val")
    dataset = UltraChat(val_data, TiktokenTokenizer("gpt2"), 1024)
    q, r = dataset[0]
    print(len(q), len(r))
    print(q[:15], r[:15])
    print(q[-10:], r[-10:])
    q, r = dataset[3]
    print(len(q), len(r))
    print(q[:15], r[:15])
    print(q[-10:], r[-10:])
