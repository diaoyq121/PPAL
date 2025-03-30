import torch
import torch.nn as nn
import torch.nn.functional as F
from docutils.nodes import attention
from openpyxl.styles.builtins import output
from src import mynn
import clip
import torch
import pandas as pd
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


class AttentionLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size=512, num_heads=8):
        super().__init__()
        # Multi-head attention module
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.q = nn.Linear(input_size, hidden_size)
        self.kv = nn.Linear(input_size, hidden_size)

        # Layer normalization
        self.norm = nn.LayerNorm(input_size)
        self.attention_V = nn.Sequential(nn.Linear(512, 512), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())

        # Feedforward network for output projection
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.linear = nn.Linear(input_size, 1)
        self.slice_selector = nn.Linear(input_size, 1)  # Learnable slice selector


    def forward(self, inputs):
       
        q = self.q(inputs)
        k = self.kv(inputs)

        attn_output, attention_weights = self.multihead_attn(q, k, k)  # Self-attention

        # Add residual connection and normalize
        attn_output = self.norm(attn_output + inputs)
        # attn_output, topk_indices = torch.topk(attn_output, k=32, dim=1)
        # slice_scores = self.slice_selector(attn_output)  # Shape (batch_size, seq_len, 1)
        # slice_mask = F.softmax(slice_scores,dim=-1)  # Binary mask (0 or 1) based on the learned slice scores
        #
        # # Apply the slice mask to the inputs, zero out unwanted slices
        # attn_output = inputs * slice_mask.float()

        # Feedforward projection

        output = self.fc(attn_output)

        # Add another residual connection and normalize
        # output = self.norm(output + attn_output)
        batch_size, seq_len, _ = output.size()
        A_V = self.attention_V(output)
        A_U = self.attention_U(output)
        A = self.linear(A_V * A_U)
        # A = torch.transpose(A, 1, 0)
        A = F.softmax(A.squeeze(-1), dim=1)
        # Calculate attention scores
        # scores = self.linear(output).view(batch_size, seq_len)

        scores = output.mean(dim=-1)
        attention_weights = F.softmax(scores, dim=1) + A
        # Apply softmax to get attention weights
        # scores = self.linear(scores)

        # attention_weights = A

        # Compute the weighted mean
        weighted_mean = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)


        return weighted_mean, attention_weights

