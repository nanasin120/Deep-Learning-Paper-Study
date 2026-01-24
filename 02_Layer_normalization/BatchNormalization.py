import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNormalization(nn.Module):
    def __init__(self, hidden_size):
        super(BatchNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size)) # README에 있던 g
        self.beta = nn.Parameter(torch.zeros(hidden_size)) # README에 있던 b

        self.register_buffer('running_mean', torch.zeros(hidden_size)) # 학습중 사용한 평균 저장 버퍼
        self.register_buffer('running_var', torch.zeros(hidden_size)) # 학습중 사용한 분산 저장 버퍼

        self.momentum = 0.1 # 저장할때 한방에 저장하는것이 아닌 부드럽게 저장시킴
        self.eps = 1e-5 # 분모가 0이 되는것을 방지하기 위함

    def forward(self, a):
        batch_size, H = a.shape # 배치 정규화는 배치를 기준으로 정규화함

        if self.training: # 학습중인 경우에는 
            # 배치를 기준으로 평균을 구하기 때문에 dim=0을 함
            mean = torch.sum(a, dim=0, keepdim=True) / batch_size
            # 이것도 배치를 기주으로 분산을 구하기 때문에 dim=0을 함
            # 표준편차가 아닌 분산을 구하는 이유는 루트가 씌워지지 않아 모맨텀 저장에 더 안정적임
            var = torch.sum((a - mean) ** 2, dim = 0, keepdim = True) / batch_size

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean # 평균 저장
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var # 분산 저장
        else: # 테스트 중인 경우에는 
            mean = self.running_mean # 저장해둔 평균
            var = self.running_var # 저장해둔 분산

        # README 수식 그대로 적용
        output = self.gamma * ((a - mean) / torch.sqrt(var + self.eps)) + self.beta

        return output