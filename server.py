import os
import chess
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tokenizers import Tokenizer
import math
import torch.nn as nn

# ========== MODELLO (uguale a quello che hai allenato) ==========
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, dropout, seq_len):
        super().__init__()
        self.c_attn = nn.Linear(n_embed, 3*n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)
        self.n_head = n_head
        self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)).view(1,1,seq_len,seq_len))
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.c_attn(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B,T,self.n_head,hs).transpose(1,2)
        k = k.view(B,T,self.n_head,hs).transpose(1,2)
        v = v.view(B,T,self.n_head,hs).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(hs))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, dropout, seq_len)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(nn.Linear(n_embed, 4*n_embed), nn.GELU(), nn.Linear(4*n_embed, n_embed), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyChessGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(seq_len, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, dropout, seq_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    def forward(self, idx, targets=None):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

# ========== CARICA MODELLO ==========
VOCAB_SIZE = 8192
SEQ_LEN = 256
DEVICE = torch.device('cpu')

print("Carico il tokenizer...")
tokenizer = Tokenizer.from_file("chess_tokenizer.json")
print("Carico il modello...")
model = TinyChessGPT(VOCAB_SIZE, SEQ_LEN, 256, 8, 6, 0.15).to(DEVICE)
model.load_state_dict(torch.load("chess_model.pt", map_location=DEVICE))
model.eval()
print("✅ Modello pronto!")

def scegli_mossa_legale(board, temperature=0.8):
    mosse_giocate = " ".join([str(m) for m in board.move_stack])
    prompt = "[BOS] " + mosse_giocate
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=DEVICE)
        logits, _ = model(input_ids)
        logits = logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
    sorted_indices = torch.argsort(probs, descending=True)
    for token_id in sorted_indices:
        mossa_testuale = tokenizer.decode([token_id]).strip()
        try:
            mossa = board.parse_san(mossa_testuale)
            if mossa in board.legal_moves:
                return mossa
        except:
            continue
    return np.random.choice(list(board.legal_moves))

# ========== SERVER FLASK ==========
app = Flask(__name__)
CORS(app)

@app.route('/move', methods=['POST'])
def get_move():
    data = request.get_json()
    fen = data['fen']
    temperature = data.get('temperature', 0.8)
    board = chess.Board(fen)
    mossa = scegli_mossa_legale(board, temperature)
    return jsonify({'move': board.san(mossa)})

@app.route('/')
def index():
    return "✅ TinyChessGPT backend is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
