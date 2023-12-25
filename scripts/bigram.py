import torch
import torch.nn as nn
import torch.nn.functional as F


# hyperparameters
# ----------------------------
batch_size = 128  # number of independent sequences to process
block_size = 8  # maximum length of the sequence at a given time
max_iters = 3000  # number of iterations to train for
eval_interval = 300  # number of iterations after which to run evaluate
eval_iters = 200  # number of iterations to run the model for evaluation on train and test batches.
learning_rate = 1e-2  # learning rate for the optimizer
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # device to use for training
# ----------------------------

# load the dataset
with open("./shakespear_input.txt", "r") as f:
    text = f.read()

print("Length of the dataset in characters:", len(text))

# All unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("All unique characters in the text:")
print("".join(chars))  # First Character is \n, hence the empty line
print("The size of the vocabulary:", vocab_size)

# Dictionary that maps characters to integers
s2i = {ch: i for i, ch in enumerate(chars)}
# Dictionary that maps integers to characters
i2s = {i: ch for i, ch in enumerate(chars)}

# enoceder and decoder
encode = lambda s: [s2i[ch] for ch in s]
decode = lambda x: "".join([i2s[i] for i in x])

# enocde the text
data = torch.tensor(encode(text), dtype=torch.long)

# Train and Test split 90%-10%
train_data = data[: int(0.9 * len(data))]
test_data = data[int(0.9 * len(data)) :]
print("Length of the train dataset:", len(train_data))
print("Length of the test dataset:", len(test_data))


# Generate batch
def get_batch(split, batch_size, block_size):
    # generate small batch of data of input:x and target:y
    data = train_data if split == "train" else test_data
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # size must be tuple hence (bs,)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x, y


# Model
class BiGramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(
            idx
        )  # (batch, seq_len) -> (batch, seq_len, emb_dim(vocab_size))

        if targets is None:
            loss = None
        else:
            batch_size, seq_len, emb_dim = logits.shape
            logits = logits.view(
                batch_size * seq_len, emb_dim
            )  # (batch, seq_len, emb_dim) -> (batch*seq_len, emb_dim)
            targets = targets.view(batch_size * seq_len)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (batch, seq_len) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction logits
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (batch, emb_dim)
            probs = torch.softmax(logits, dim=-1)  # (batch, emb_dim)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq_len+1)

        return idx


model = BiGramLanguageModel(vocab_size).to(device)


@torch.no_grad()
def evaluate(eval_iters):
    model.eval()
    losses = {}
    for split in ["train", "test"]:
        losses[split] = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(split, batch_size, block_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[split] += loss.item()
        losses[split] /= eval_iters
    model.train()
    return losses


# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for i in range(max_iters):
    # evaluate model every eval_interval iterations
    if i % eval_interval == 0:
        losses = evaluate(eval_iters)
        print(
            f"iteration {i}: train_loss = {losses['train']:.4f}, test_loss = {losses['test']:.4f}"
        )

    # get batch
    x, y = get_batch("train", batch_size, block_size)
    x, y = x.to(device), y.to(device)

    # forward pass
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate text
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(
    decode(
        model.generate(context, max_new_tokens=500)[
            0
        ].tolist()  # Take the first seq in batch
    )
)
