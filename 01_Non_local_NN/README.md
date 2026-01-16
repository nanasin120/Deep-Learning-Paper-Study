# Non-local operation

# 소개

컨볼루션 연산이나 재귀 연산은 입력의 전체가 아닌 부분 부분을 처리한다.

non-local 연산은 한번에 입력 전체를 처리하는 연산이다.

## 장거리 의존성

장거리 의존성은 멀리 떨어져 있는 데이터들의 관계성이다.

이 값이 높으면 서로 관계가 있는 데이터라는 의미이다.

# Non-local operation 그림

![image.png](image.png)
> *그림 출처: Wang et al., "Non-local Neural Networks", CVPR 2018*

여기서 X는 입력 데이터, ⊗은 행렬 곱, ⊕은 원소별 합을 나타낸다.

처음에 X는 channel이 1024인 상태로 들어온다.

이때 가중치 $W_θ$와 $W_ϕ$를 지나고 나서는 channel이 512로 줄어든다. 이는 bottleneck에서 가져온 것이다.

계산량을 절반으로 줄여준다.

이는 g에서도 마찬가지이다.

# 식

non-local operation을 식으로 쓰면 다음과 같다.

$\mathbf{y_i} = \frac{1}{\mathcal{C}(\mathbf{x})} \sum\limits_{\forall j} f(\mathbf{x}_i, \mathbf{x}_j)g(\mathbf{x}_j)$ (1)

$f(x_i, x_j)=e^{θ(x_i)^Tϕ(𝐱_j)}.$ (2)

$θ(x_i)=W_θx_i$ (3)

$ϕ(𝐱_j)=W_ϕx_j$ (4)

$𝒞(x)=\sum\limits_{\forall j} f(x_i, x_j)$ (5)

$g(x_j)=W_gx_j$ (6)

$z_i = W_zy_i + x_i$ (7)

보면 3번과 4번 식의 가중치가 다르다. 이는 서로 다른 데이터도 비슷하게 값을 맞춰준다.

5번 식은 모든 $f$함수의 출력을 더하는걸 볼 수 있다. 이는 softmax와 형태가 같아지게 한다.

6번 식은 $g$함수를 보여준다. 가중치 $W_g$와 곱함으로서 데이터의 특징을 뽑아낸다.

7번 식의 $W_z$는 마지막 원소별 합으로 512로 줄였던 channel을 1024로 다시 돌려준다.

---
## 참고 문헌 (References)
* **Paper**: Wang, X., Girshick, R., Gupta, A., & He, K. (2018). [Non-local Neural Networks](https://arxiv.org/abs/1711.07971). *Proceedings of the IEEE conference on computer vision and pattern recognition*.
