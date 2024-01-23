import torch
from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, vocab_size=1000, embed_dim=300, max_seq_len=100, num_heads=8, dropout_rate=0.1):
        self.tokenizer = None
        self.pos_encoder = None
        self.embedder = None
        self.encoders = [EncoderBlock(embed_dim, num_heads, dropout_rate) for _ in range(6)]

    def forward(self, words: torch.Tensor):
        tok = self.tokenizer(words)
        emb = self.embedder(tok)
        pos_e = self.pos_encoder(emb)
        enc = pos_e
        for encoder in self.encoders:
            enc = encoder(enc)

        return enc

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
        

class EncoderBlock(nn.Module):
    
        def __init__(self, embed_dim=300, num_heads=8, dropout_rate=0):
            self.mha = MultiHeadAttention()
            self.ln1 = LayerNorm(embed_dim=embed_dim) # layer norm
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.ln2 = LayerNorm(embed_dim=embed_dim) # layer norm
    
        def forward(self, x):
            attn = self.mha(x, x, x)
            x = self.ln1(x + attn)
            ffn = self.ffn(x)
            x = self.ln2(x + ffn)
            return x


class MultiHeadAttention(nn.Module):

    def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask=None):
        qv_scaled = (q @ k) / q.size[-1]**0.5
        if key_mask is not None:
            # TODO: implement key_mask on qv_scaled
            pass
        # Do softmax for each query/token
        probs = F.softmax(qv_scaled, dim=-1)
        return probs @ v
    
    def __init__(self, embed_dim=300, num_heads=8, dropout_rate=0):
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads

    def forward(self, q, k, v, key_mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # Split q, k, v into num_heads
        q = torch.stack(q.split(q.size[-1] // self.num_heads, -1), dim=0)
        k = torch.stack(k.split(k.size[-1] // self.num_heads, -1), dim=0)
        v = torch.stack(v.split(v.size[-1] // self.num_heads, -1), dim=0)
        # Do scaled dot product attention
        scaled_attention = self.scaled_dot_product_attention(q, k, v, key_mask)
        # Concatenate all the heads back together
        # ????
        concat_attention = scaled_attention.transpose(1, 2).reshape(q.size(0), -1, q.size(-1))
        # Apply a final linear layer
        output = nn.Linear(q.size(-1), q.size(-1))(concat_attention)
        return output

class MyNetwork(nn.Module):
    
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()
    
        def forward(self, words: torch.Tensor):
            x = self.transformer(words)
            return x
        

model = MyNetwork()
words = torch.rand(10, 100)
sentiment = torch.rand(10, 1)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 10
for i in range(epoch):
    optimizer.zero_grad()
    out = model(words)
    l = loss(out, torch.ones_like(out))
    l.backward()
    optimizer.step()
    print(l)    
