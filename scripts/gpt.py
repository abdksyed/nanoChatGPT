import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# hyperparameters
# ----------------------------
batch_size = 8  # number of independent sequences to process
seq_len = 512  # maximum length of the sequence at a given time
max_iters = 5000  # number of iterations to train for
eval_interval = (
    max_iters // 10
)  # 500 # number of iterations after which to run evaluate
eval_iters = 200  # number of iterations to run the model for evaluation on train and test batches.
learning_rate = 3e-4  # learning rate for the optimizer
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # device to use for training

emb_dim = 768  # embedding dimension
num_heads = 12  # number of heads in the multi-head attention layer
num_blocks = 12  # number of blocks in the transformer
dropout = 0.2  # dropout probability
# ----------------------------


# # hyperparameters for local testing
# # ----------------------------
# batch_size = 32  # number of independent sequences to process
# seq_len = 8  # maximum length of the sequence at a given time
# max_iters = 5000  # number of iterations to train for
# eval_interval = (
#     max_iters // 10
# )  # 500 # number of iterations after which to run evaluate
# eval_iters = 200  # number of iterations to run the model for evaluation on train and test batches.
# learning_rate = 1e-3  # learning rate for the optimizer
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )  # device to use for training

# emb_dim = 32  # embedding dimension
# num_heads = 4  # number of heads in the multi-head attention layer
# num_blocks = 3  # number of blocks in the transformer
# dropout = 0.2  # dropout probability
# # ----------------------------

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
def get_batch(split, batch_size, seq_len):
    # generate small batch of data of input:x and target:y
    data = train_data if split == "train" else test_data
    ix = torch.randint(
        len(data) - seq_len, (batch_size,)
    )  # size must be tuple hence (bs,)
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    return x, y


# Self Attention
class Head(nn.Module):
    "Single Head of Self Attention"

    def __init__(self, emb_dim: int, head_size: int, seq_len: int, dropout:float):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        # tril matrix for masking is not a learnable parameter
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        k = self.key(x)  # (batch, seq_len, emb_dim) -> (batch, seq_len, head_size)
        q = self.query(x)  # (batch, seq_len, emb_dim) -> (batch, seq_len, head_size)
        v = self.value(x)  # (batch, seq_len, emb_dim) -> (batch, seq_len, head_size)

        # compute attention weights
        # (batch, seq_len, head_size) @ (batch, head_size, seq_len) -> (batch, seq_len, seq_len)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # mask the weights
        weights = weights.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        # apply softmax
        weights = torch.softmax(weights, dim=-1)
        # apply dropout
        weights = self.dropout(weights)
        # apply attention
        out = (
            weights @ v
        )  # (batch, seq_len, seq_len) @ (batch, seq_len, head_size) -> (batch, seq_len, head_size)

        return out


# Multi Head Attention
class MultiHeadAttention(nn.Module):
    "Multiple heads od Self-Attention in Parallel"

    def __init__(self, emb_dim:int, head_size:int, seq_len:int, num_heads:int, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(emb_dim, head_size, seq_len, dropout) for _ in range(num_heads)])
        # Projection Layer to bring from residual path back to main path
        self.projection = nn.Linear(head_size*num_heads, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (batch, seq_len, emb_dim)
        # each head is (batch, seq_len, head_size)
        # we concatenate all heads together and project back to emb_dim
        # [(batch, seq_len, emb_dim//num_heads), ..., (batch, seq_len, emb_dim//num_heads)] -> (batch, seq_len, emb_dim)
        out = torch.cat([head(x) for head in self.heads], dim=-1) 
        out = self.dropout(self.projection(out))
        return out


# Position-wise Feed Forward Network
class FeedForwardNetowrk(nn.Module):
    "Single Layer Feed Forward Network with non-linearity"

    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim*4)
        self.relu = nn.ReLU()
        # Projection Layer to bring from residual path back to main path
        self.linear2 = nn.Linear(emb_dim*4, emb_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return x


# Decoder Block
class Block(nn.Module):
    "Single Block of Transformer"

    def __init__(self, emb_dim: int, seq_len: int, num_heads: int, dropout: float):
        super().__init__()
        head_size = emb_dim // num_heads
        self.sa_head = MultiHeadAttention(emb_dim, head_size, seq_len, num_heads, dropout)
        self.ffn = FeedForwardNetowrk(emb_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.sa_head(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Model
class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim, num_heads, num_blocks, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.postion_embedding_table = nn.Embedding(seq_len, emb_dim)

        self.blocks = nn.Sequential(
            *[Block(emb_dim, seq_len, num_heads, dropout) for _ in range(num_blocks)]
        )

        self.norm_final = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (batch, seq_len)

        tok_emb = self.token_embedding_table(
            idx
        )  # (batch, seq_len) -> (batch, seq_len, emb_dim)

        pos_emb = self.postion_embedding_table(
            torch.arange(idx.shape[1], device=device)
        )  # (seq_len, emb_dim)

        x = (
            tok_emb + pos_emb
        )  # (batch, seq_len, emb_dim) + (seq_len, emb_dim) -> (batch, seq_len, emb_dim)

        x = self.blocks(x)  # (batch, seq_len, emb_dim)

        x = self.norm_final(x)  # (batch, seq_len, emb_dim)

        logits = self.lm_head(
            x
        )  # (batch, seq_len, emb_dim) -> (batch, seq_len, vocab_size)

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
            # the idx should never be longer than seq_len
            # Take the last `seq_len` tokens for each sequence in the batch
            idx_cropped = idx[:, -seq_len:]  # (batch, seq_len)

            # get the prediction logits
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :]  # (batch, emb_dim)
            probs = torch.softmax(logits, dim=-1)  # (batch, emb_dim)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq_len+1)

        return idx


model = GPT(
    vocab_size=vocab_size,
    seq_len=seq_len,
    emb_dim=emb_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    dropout=dropout,
).to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


@torch.no_grad()
def evaluate(eval_iters):
    model.eval()
    losses = {}
    for split in ["train", "test"]:
        losses[split] = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(split, batch_size, seq_len)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[split] += loss.item()
        losses[split] /= eval_iters
    model.train()
    return losses


# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for i in range(1, max_iters + 1):
    # evaluate model every eval_interval iterations
    if i % eval_interval == 0:
        losses = evaluate(eval_iters)
        print(
            f"iteration {i}: train_loss = {losses['train']:.4f}, test_loss = {losses['test']:.4f}"
        )

    # get batch
    x, y = get_batch("train", batch_size, seq_len)
    x, y = x.to(device), y.to(device)

    # forward pass
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), './weights/gpt.pth')

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
