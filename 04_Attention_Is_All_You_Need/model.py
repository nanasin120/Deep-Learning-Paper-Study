import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from blocks import PositionalEncoding, MultiHead, FeedForwardNetwork

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.h = 8
        self.d_model = 512
        self.vocab_size = vocab_size

        # nn.Embedding(단어의 개수, 임베딩할 벡터 차원)
        self.multi_head_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, Inputs, mask):
        # 기존에 [배치 사이즈, 문장 길이] 이었던게 [배치 사이즈, 문장 길이, d_model]로 변함
        # [배치 사이즈, 문장 길이, d_model]이 반환됨
        after_multi_head = self.multi_head_attention(Inputs, Inputs, Inputs, mask=mask)

        # Add & Norm 처리 해줌
        after_multi_head = self.layer_norm_1(after_multi_head + Inputs)

        after_ffn = self.FFN(after_multi_head)

        after_ffn = self.layer_norm_2(after_multi_head + after_ffn)

        return after_ffn

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.h = 8
        self.d_model = 512
        self.vocab_size = vocab_size
        # nn.Embedding(단어의 개수, 임베딩할 벡터 차원)

        self.masked_multi_head_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_0 = nn.LayerNorm(self.d_model)

        self.multi_head_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)

        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, Inputs, Outputs, enc_mask, dec_mask):

        # masked Multi head
        after_masked_multi_head = self.masked_multi_head_attention(Outputs, Outputs, Outputs, dec_mask)
        after_masked_multi_head = self.layer_norm_0(after_masked_multi_head + Outputs)

        # [배치 사이즈, 문장 길이, d_model]이 반환됨
        after_multi_head = self.multi_head_attention(after_masked_multi_head, Inputs, Inputs, enc_mask)

        # Add & Norm 처리 해줌
        after_multi_head = self.layer_norm_1(after_multi_head + after_masked_multi_head)

        after_ffn = self.FFN(after_multi_head)
        after_ffn = self.layer_norm_2(after_multi_head + after_ffn)

        return after_ffn


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super(Transformer, self).__init__()
        self.n = 6
        self.d_model = 512
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.output_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.position_encoding = PositionalEncoding(self.d_model, 128)

        self.encoder_stack = nn.ModuleList([Encoder(vocab_size) for _ in range(self.n)])
        self.decoder_stack = nn.ModuleList([Decoder(vocab_size) for _ in range(self.n)])

        self.linear = nn.Linear(self.d_model, vocab_size)
        
        self.input_embedding.weight = self.output_embedding.weight
        self.linear.weight = self.output_embedding.weight

    def forward(self, Inputs, Outputs):
        enc_mask = (Inputs != 0).unsqueeze(1).unsqueeze(2)
        dec_mask = (Outputs != 0).unsqueeze(1).unsqueeze(2)

        after_input_embedding = self.input_embedding(Inputs)
        after_input_embedding = self.position_encoding(after_input_embedding)
        
        enc_out = after_input_embedding
        for layer in self.encoder_stack:
            enc_out = layer(enc_out, enc_mask)

        after_output_embedding = self.output_embedding(Outputs)
        after_output_embedding = self.position_encoding(after_output_embedding)

        dec_out = after_output_embedding
        for layer in self.decoder_stack:
            dec_out = layer(enc_out, dec_out, enc_mask, dec_mask)

        output = torch.softmax(self.linear(dec_out), dim=-1)

        return output