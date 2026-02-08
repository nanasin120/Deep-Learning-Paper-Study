import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def Attention(Q, K, V, d_k, mask):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, V)

    return output

class MultiHead(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHead, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        # 입력으로 들어오는 Q, K, V 모두 [batch_size, sequence_len, d_model]
        # 여기서 d_model을 h개로 나눠 나눈것들에 matmul을 하고 attention을 한뒤 concat하고 다시 matmul해야함
        # 이 과정을 for문을 사용하지 않고 할거임
        
        # 배치 사이즈와 문장의 길이를 사용할거임
        batch_size, sequence_len, d_model = Q.shape

        # Linear로 한번에 가중치 연산을 해준뒤 self.h로 뒷부분을 나눠주고 1번과 2번을 바꿔줌
        # 이렇게 되면 [배치 사이즈, self.h, 문장의 길이, self.d_k]로 나뉘게됨
        # 이건 마치 matmul을 한뒤 self.h로 나눈것과 같음을 알 수 있음
        nQ = self.WQ(Q).view(batch_size, sequence_len, self.h, self.d_k).transpose(1, 2)
        nK = self.WK(K).view(batch_size, sequence_len, self.h, self.d_k).transpose(1, 2)
        nV = self.WV(V).view(batch_size, sequence_len, self.h, self.d_k).transpose(1, 2)

        # 그후 head를 구해줌, 이 head는 concat할 필요가 없음 이미 다 붙어있으니
        head = Attention(nQ, nK, nV, self.d_k, mask)

        # [배치 사이즈, self.h, 문장 길이, self.d_k]인거를
        # [배치 사이즈, 문장 길이, self.h, self.d_k]로 바꿔주고
        # [배치 사이즈, 문장 길이, self.h * self.d_k = self.d_model]로 바꿔줌
        head = head.transpose(1, 2).contiguous().view(batch_size, sequence_len, self.d_model)
        
        # 마지막으로 WO와 곱하면 끝
        output = self.WO(head)

        # [배치 사이즈, 문장 길이, self.d_model]으로 반환됨
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        

    def forward(self, x):
        x = torch.relu(self.linear1(x))

        output = self.linear2(x)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_len):
        super(PositionalEncoding, self).__init__()

        # pe = [최대 문장 길이, d_model]
        pe = torch.zeros(max_sequence_len, d_model)
        # position = 0 ~ 최대 문장 길이, 여기에 1차원을 추가해줌
        # [
        # [0], 
        # [1], 
        # [2], 
        # ...] 이런 상태임
        position = torch.arange(0, max_sequence_len, dtype=torch.float).unsqueeze(1)

        # 그냥 10,000 ^ (2i / d_model)을 할수도 있지만 exp를 이용해 연산량이 쉽게 할 수 있음
        # 원래 값 10,000 ^ (2i / d_model)에 로그를 씌우고 다시 exp를 씌우면
        # exp(-ln(10,000) * 2i / d_model) 이 됨, 이건 원래랑 같은 식임
        # torch.arrange(0, d_model, 2)는 0, 2, 4, 6, ... 이렇게 2씩 증가하는거고
        # 뒤에는 위에서 계산한대로 진행됨
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 각자 자리에 맞는 것들로 계산해줌
        pe[:, 0::2] = torch.sin(position * div) # 짝수
        pe[:, 1::2] = torch.cos(position * div) # 홀수

        # 이제 0번째 차원을 만들어줌
        # [1, 최대 문장 길이, d_model]
        pe = pe.unsqueeze(0)

        # self.pe가 아니라 register_buffer를 사용하면 연산도 빨라지고 좋음
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x는 [배치 사이즈, 문장 길이, d_model]
        # pe는 [전부, 문장의 길이만큼, 전부]를 의미한다.
        x = x + self.pe[:, :x.shape[1], :]
        return x