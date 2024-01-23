import torch
from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, vocab_size=1000, embed_dim=300, max_seq_len=100, num_heads=8, dropout_rate=0.1):
        self.tokenizer = None
        self.pos_encoder = None
        self.embedder = None
        self.w_q = 
        self.w_k = 
        self.w_v = 

    def scaled_dot_product_attention(q, k, v, key_mask=None):

        qv_scaled = (q @ k) / q.size[-1]**0.5
        if key_mask is not None:
            # TODO: implement key_mask on qv_scaled
            pass
        # Do softmax for each query/token
        probs = F.softmax(qv_scaled, dim=-1)
        return probs @ v


    # def multihead_attention(q, k, v, key_masks, num_heads=8, dropout_rate=0):
    #     # Split q, k, v into num_heads
    #     q = torch.stack(q.split(q.size[-1] // num_heads, -1), dim=0)
    #     k = torch.stack(k.split(k.size[-1] // num_heads, -1), dim=0)
    #     v = torch.stack(v.split(v.size[-1] // num_heads, -1), dim=0)
    #     # Do scaled dot product attention
    #     scaled_attention = scaled_dot_product_attention(q, k, v, key_masks)
    #     # Concatenate all the heads back together
    #     concat_attention = scaled_attention.transpose(1, 2).reshape(q.size(0), -1, q.size(-1))
    #     # Apply a final linear layer
    #     output = nn.Linear(q.size(-1), q.size(-1))(concat_attention)
    #     return output

    def forward(self, words: torch.Tensor):
        tok = self.tokenizer(words)
        emb = self.embedder(tok)
        pos_e = self.pos_encoder(emb)
        q = pos_e @ self.w_q
        k = pos_e @ self.w_k
        v = pos_e @ self.w_v
        return self.scaled_dot_product_attention(q, k, v)
    

class MyNetwork(nn.Module):
    
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()
            self.linear = nn.Linear(300, 1, bias=False)
    
        def forward(self, words: torch.Tensor):
            x = self.transformer(words)
            out = self.linear(x)
            return out
        

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
