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
VOCAB_SIZE = 2935   # 🔧 Controlla che sia il valore corretto (vedi nota)
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

def get_game_over_reason(board):
    if board.is_checkmate(): return 'CHECKMATE'
    if board.is_stalemate(): return 'STALEMATE'
    if board.is_insufficient_material(): return 'INSUFFICIENT_MATERIAL'
    if board.is_seventyfive_moves(): return 'SEVENTYFIVE_MOVES'
    if board.is_fivefold_repetition(): return 'FIVEFOLD_REPETITION'
    return 'UNKNOWN'

# ========== SERVER FLASK ==========
app = Flask(__name__)
CORS(app)

@app.route('/legal_moves', methods=['POST'])
def legal_moves():
    data = request.get_json()
    board = chess.Board(data['fen'])
    moves = [move.uci() for move in board.legal_moves]
    return jsonify({'moves': moves})

@app.route('/validate', methods=['POST'])
def validate_move():
    data = request.get_json()
    board = chess.Board(data['fen'])
    uci = data['uci']
    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            san = board.san(move)   # va calcolato PRIMA di push()
            board.push(move)
            return jsonify({
                'valid': True,
                'fen': board.fen(),
                'san': san,
                'in_check': board.is_check(),
                'game_over': board.is_game_over(),
                'result': board.result() if board.is_game_over() else None,
                'reason': get_game_over_reason(board) if board.is_game_over() else None
            })
        else:
            return jsonify({'valid': False})
    except:
        return jsonify({'valid': False})

@app.route('/move', methods=['POST'])
def get_move():
    data = request.get_json()
    board = chess.Board(data['fen'])
    temperature = data.get('temperature', 0.8)
    mossa = scegli_mossa_legale(board, temperature)
    san = board.san(mossa)   # va calcolato PRIMA di push()
    board.push(mossa)
    return jsonify({
        'status': 'game_over' if board.is_game_over() else 'ok',
        'fen': board.fen(),
        'uci': mossa.uci(),
        'san': san,
        'in_check': board.is_check(),
        'result': board.result() if board.is_game_over() else None,
        'reason': get_game_over_reason(board) if board.is_game_over() else None
    })

@app.route('/')
def index():
    return "✅ TinyChessGPT backend is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)