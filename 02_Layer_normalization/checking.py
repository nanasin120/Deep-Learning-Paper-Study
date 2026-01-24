import torch
import torch.nn as nn
import torch.nn.functional as F
from LayerNormalization import LayerNormalization
from BatchNormalization import BatchNormalization

print("테스트 데이터")
data = torch.randn((5, 8))
print(data)

BN = BatchNormalization(8)
LN = LayerNormalization(8)

print("학습중 배치 정규화 이후 데이터")
data_bn = BN(data)
print(data_bn)
print("학습중 래이어 정규화 이후 데이터")
data_ln = LN(data)
print(data_ln)

BN.eval()
LN.eval()

print("테스트중 배치 정규화 이후 데이터")
data_bn = BN(data)
print(data_bn)
print("테스트중 래이어 정규화 이후 데이터")
data_ln = LN(data)
print(data_ln)
