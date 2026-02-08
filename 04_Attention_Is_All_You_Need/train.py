import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Transformer

batch_size = 64
sequence_len = 128
vocab_size = 10000

inputs = torch.randint(0, vocab_size, (batch_size, sequence_len))
outputs = torch.randint(0, vocab_size, (batch_size, sequence_len))

transformer = Transformer(vocab_size)

output = transformer(inputs, outputs)

print(output.shape)