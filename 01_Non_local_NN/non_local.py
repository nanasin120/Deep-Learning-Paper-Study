import torch
import torch.nn as nn
import torch.nn.functional as F

class Non_local_block(nn.Module):
    def __init__(self, in_channels):
        super(Non_local_block, self).__init__()
        
        self.in_channel = in_channels
        self.theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, bias=False)
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, bias=False)
        self.W_z = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, bias=False)

        nn.init.constant_(self.W_z.weight, 0)

    def forward(self, x):
        # 배치 사이즈, 채널, 높이, 너비
        batch, channel, height, width = x.shape

        print(f'x shape : {x.shape}')

        # g에 넣어서 값을 구해준뒤 view를 이용해 h*w를 해준다.
        # 처음 self.g에 넣고 나서는 (배치 사이즈, 채널//2, 높이, 너비)로 나오고
        # view를 통해 (배치 사이즈, 채널//2, 높이 * 너비)로 교체된다.
        # 식(6)에 해당한다.
        g = self.g(x)
        g = g.view(batch, self.in_channel//2, -1)

        print(f'g shape : {g.shape}')

        # theta와 phi도 똑같이 해준다.
        # 식(3)과 식(4)번이다.
        # 이것들도 (배치 사이즈, 채널//2, 높이 * 너비)로 교체된다.
        x_theta = self.theta(x)
        x_theta = x_theta.view(batch, self.in_channel//2, -1)
        print(f'x_theta shape : {x_theta.shape}')

        x_phi = self.phi(x)
        x_phi = x_phi.view(batch, self.in_channel//2, -1)
        print(f'x_phi shape : {x_phi.shape}')

        # 행렬의 곱을 위해 permute를 사용하여 transpose를 해준다.
        # 이렇게 하면 (배치 사이즈, 높이 * 너비, 채널)로 변한다.
        # 이제 matmul을 해주면 (배치 사이즈, 높이 * 너비, 높이 * 너비)로 나온다.
        x_theta_T = x_theta.permute(0, 2, 1)
        f = torch.matmul(x_theta_T, x_phi)
        # 마지막으로 exp에 넣어주면 식(2)가 완성된다.
        f = torch.exp(f)
        
        # f안의 모든 값을 더한 c이다. 
        # (배치 사이즈, 높이 * 너비, 1)로 나온다.
        # 식(5)에 해당한다.
        c = torch.sum(f, dim = -1, keepdim=True)

        # 그 후 f의 모든 값을 c로 나눔으로서 softmax를 취해준다.
        f = f / c
        print(f'f shape : {f.shape}')

        # 이제 f와 g를 행렬 곱해주면 식(1)이 완성된다.
        # g는 (배치 사이즈, 채널//2, 높이 * 너비)이고
        # g를 permute를 통해 (배치 사이즈, 높이 * 너비, 채널//2)로 바꿔준다.
        g = g.permute(0, 2, 1)
        # f는 (배치 사이즈, 높이 * 너비, 높이 * 너비)이다.
        # 이 둘을 matmul해주면 (배치 사이즈, 높이 * 너비, 채널//2)로 나온다.
        y = torch.matmul(f, g)

        # 이제 다시 y를 (배치 사이즈, 채널//2, 높이 * 너비)로 바꿔준다.
        # 그리고 (배치 사이즈, 채널//2, 높이, 너비)로 바꿔준다.
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch, self.in_channel//2, height, width)
        
        # 그 후 self.W_z를 통해 채널//2를 채널로 다시 돌리고
        # x와 원소별 합을 해주면 식(1)이 완성된다.
        z = self.W_z(y) + x
        print(f'z shape : {z.shape}')

        return z

if __name__ == '__main__':
    x = torch.randn(32, 1024, 32, 32)
    non_local = Non_local_block(1024)
    z = non_local(x)

    print(f'input shape : {x.shape}')
    print(f'ouput shape : {z.shape}')