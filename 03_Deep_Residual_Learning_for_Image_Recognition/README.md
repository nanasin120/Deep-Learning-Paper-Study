# 간단 소개

잔차 학습을 통해 최적화와 네트워크 깊이에서 이점을 얻을 수 있다.

BottleNeck을 통해 연산량을 줄일 수 있다.

# 잔차 학습

논문에서는 

잔차 매핑을 최적화하는것이 원래의 참조되지 않은 매핑을 최적화하는것보다 쉽다란 가설을 세웠다.

잔차는 실제 값과 근사 값의 차이이다.

‘매핑’(mapping)은 입력과 출력 사이의 대응 관계이다. 

우리는 어떠한 매핑를 목표로 네트워크를 학습시킨다. 이 매핑를 ‘기저 매핑’라 한다.

기저매핑을 $\mathcal{H}(x)$으로 정의하고, 네트워크 속, 예를 들어 [Conv2D, BatchNorm, ReLU]로 이루어진 레이어는  다음과 같은 매핑을 학습하게 한다.

$\mathcal{F}(x) =$  Conv2D, BatchNorm, ReLU

$\mathcal{F}(x):= \mathcal{H}(x) -x$

$\mathcal{H}(x)$는 목표로 하는 기저매핑, 즉 실제값과 가장 유사한 값을 출력하는 매핑이다.

이렇게 되면 목표로 하는 기저 매핑은 $F(x)+x$로 재구성 된다.

이를 위해 ‘지름길 연결’(shortcut connection)을 이용한다.

논문에서는 정의를 다음과 같이 한다.

$y=\mathcal{F}(x, \{W_i\})+x$

$x, y$는 레이어의 입력 출력 벡터이다.

여기서  $\mathcal{F}(x, \{W_i\})$는 학습되어야할 잔차 매핑을 의미한다.

$\mathcal{F}+x$ 연산은 지름길 연결과 원소별 덧셈을 통해 수행된다.

$x$와 $\mathcal{F}$의 차원은 같아야 한다. 

만약 그렇지 않은 경우 (예: 입력/출력이 바뀐 경우) 우리는 차원을 맞추기 위한 지름길 연결로 $W_s$ 선형 투영을 수행할 수 있다.

$y=\mathcal{F}(x, \{W_i\})+W_sx$

이를 통해 CNN 연산 또한 수행할 수 있다.

# BottleNeck

BottleNeck은 차원을 축소하고 다시 복원함으로서 연산량을 줄여준다.

논문에서는 1 x 1, 3 x 3, 1 x 1의 합성곱으로 이루어진 3개의 레이어를 이용했다.

1 x 1 레이어는 차원을 축소하고 다시 복원하는 역할을 한다.

3 x 3레이어는 더 적은 연산량을 갖게 된다.

BottleNeck에서는 항등 지름길 (identity shortcut)이 특히 더 중요하다.

이를 위에서 말한 투영으로 대체할경우 추가 가중치 행렬이 필요하고 이렇게 되면 연산량이 늘어나

BottleNeck을 사용한 이유가 없어진다.

## 참고 문헌 (References)

- **Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. arXiv preprint arXiv:1512.03385. (Presented at CVPR 2016)