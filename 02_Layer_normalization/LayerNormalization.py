import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size)) # README에 있던 g
        self.beta = nn.Parameter(torch.zeros(hidden_size)) # READMD에 있던 b
        self.f = nn.ReLU() # README에 있던 f() 함수


    def forward(self, a):
        batch_size, H = a.shape # 레이어 정규화는 배치가 아닌 입력 데이터를 기준으로 정규화
        
        mean = torch.sum(a, dim = -1, keepdim=True) / H # 입력 데이터를 기준으로 평균 구하기

        eps = 1e-5 # std는 분모에 들어가니 혹여라도 0이 되는것을 방지하기 위한 매우 작은 값
        std = torch.sqrt(torch.sum((a-mean)**2, dim=-1, keepdim=True) / H + eps) # 표준편차 구하기
        
        h = self.f(self.gamma * (a - mean) / std + self.beta) # README 식 그대로 수식 작성

        return h