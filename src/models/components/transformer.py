import math

import torch
from torch import nn
from torchvision import models


class Transformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        pos_dropout,
        trans_dropout,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        self.pos_dropout = pos_dropout
        self.trans_dropout = trans_dropout
        self.vocab_size = len(tokenizer.chars)
        self.embed_tgt = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_enc = PositionalEncoder(self.d_model, self.pos_dropout, self.max_seq_length)
        self.transformer = nn.Transformer(
            self.d_model,
            self.nhead,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.dim_feedforward,
            self.trans_dropout,
        )
        self.fc = nn.Linear(self.d_model, self.vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Shape:
            - src: (W, B, C)
            - tgt: (T, B)
            - src_key_padding_mask: (B, S)
            - tgt_key_padding_mask: (B, T)
            - memory_key_padding_mask: (B, S)
            -> output: (B, T, E)
        """

        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(src.device)
        src = self.pos_enc(src * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = output.transpose(0, 1)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        mask = (
            torch.triu(
                torch.ones(
                    length,
                    length,
                )
            )
            == 1
        ).transpose(0, 1)
        mask = (
            mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward_encoder(self, src):
        src = self.pos_enc(src * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src)
        return memory

    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return self.fc(output), memory

    def expand_memory(self, memory, beam_size):
        return memory.repeat(1, beam_size, 1)

    def get_memory(self, memory, i):
        return memory[:, [i], :]


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pos_encode = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pos_encode[:, 0::2] = torch.sin(pos * div_term)
        pos_encode[:, 1::2] = torch.cos(pos * div_term)
        pos_encode = pos_encode.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encode", pos_encode)

    def forward(self, x):
        x = x + self.pos_encode[: x.size(0), :]
        return self.dropout(x)
