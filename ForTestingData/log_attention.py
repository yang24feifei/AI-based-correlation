import torch
import torch.nn as nn
import torch.nn.functional as F

class LogAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,  param_vocab_size, param_embed_dim, batch_first= False):
        super(LogAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.param_dim = param_dim
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Character embedding for parameter encoding
        self.char_embedding = nn.Embedding(param_vocab_size, param_embed_dim)
        self.param_encoder = nn.Linear(param_embed_dim, self.head_dim)  #  embed_dim)

        # Final output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # # option: convert phi_p vector to scalar
        # self.bias_proj=nn.Linear(self.head_dim,1)
    
    def forward(self, x, param_chars):  # option-TF, long_log_attention):
        # print(x.size())
        # print(param_chars)
        
        seq_len = x.shape[0]

        # Compute Q, K, V: (seq_len, num_heads, head_dim)
        Q = self.W_q(x).view(seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        # Compute parameter encoding using character embeddings
        param_embedded = self.char_embedding(param_chars)  # (param_len, param_embed_dim)
        phi_p = self.param_encoder(param_embedded)  # (param_len, head_dim)
        # phi_p = phi_p.view(self.num_heads, self.head_dim)  # (num_heads, head_dim)
        
        phi_p =phi_p.unsqueeze(0) 
        phi_p = phi_p.repeat(self.num_heads, 1, 1)  # (num_heads, seq_len, head_dim)
        
        phi_p_scalar = phi_p.mean(dim=-1)  # -> [num_heads, seq_len]
        # phi_p_scalar = phi_p.max(dim=-1)  # -> [num_heads, seq_len]        
        # phi_p_scalar = self.bias_proj(phi_p)  # -> [num_heads, seq_len]
        
        # Compute attention scores: (num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # print(attn_scores.shape)
        # print(phi_p.shape)
        # Add parameter bias
        attn_scores += phi_p_scalar.unsqueeze(-1) 

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention output: (num_heads, seq_len, head_dim)
        output = torch.matmul(attn_probs, V)

        # Reshape and project back: (seq_len, embed_dim)
        output = output.transpose(0, 1).contiguous().view(seq_len, self.embed_dim)
        output = self.W_o(output)

        return output