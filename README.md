손실함수

지표

상황 판단을 내리기 위한 기준을 정량화 한 것

머신러닝에서의 지표

MSE, MAE, RMSE

재현율, 정밀도, f1 score

Accuracy

신경망에서의 지표

머신러닝에서 사용하는 지표는 모두 사용

잘 맞춘 것을 지표로 삼을 것인지( score / accuracy )

잘 못맞춘 것을 지표로 삼을 것인지 ( loss / cost ) - 잘 못맞춘 것을 주로 사용한다.

좋은 신경망이란? Loss 가 적은 신경망 ( 잘 못맞춘 것이 낮은 신경망이 좋은 신경망 )

평균 제곱 오차 ( Mean Squared Error )

신경망에서의 MSE

MSE=12∑k(yk−tk)2

인간이 신경망을 공부할 때 사용하는 공부용 MSE 입니다..

yk : 신경망의 예측값

tk : 정답 레이블

# 이 함수는 의미 파악만 할 것!

def numerical_gradient(f, x):

  h = 1e-4

  grad = np.zeros_like(x) # x와 shape이 같은 0으로 채워진 배열을 만든다.

​

  for idx in range(x.size):

    # 각 x에 대한 편미분을 수행한다.

    tmp_val = x[idx]

​

    # f(x+h) 계산

    x[idx] = tmp_val + h # 목표로 하고 있는 x의 변화량을 구하기 위함

    fxh1 = f(x) # 변화량을 준 (tmp_val+h) x에 대한 전방차분 구하기

    

    # f(x-h) 계산

    x[idx] = tmp_val - h

    fxh2 = f(x)

​

    grad[idx] = (fxh1 - fxh2) / 2*h # 미분 수행

    x[idx] = tmp_val # 원래 값으로 복원 하기

  

  return grad

x = np.array([3.0, 4.0])

print("x = [3, 4] 일 때의 기울기 배열 : {}".format(numerical_gradient(function_2, x)))

​

x = np.array([1.0, 2.0])

print("x = [1, 2] 일 때의 기울기 배열 : {}".format(numerical_gradient(function_2, x)))

​

x = np.array([1.0, 1.0])

print("x = [1, 1] 일 때의 기울기 배열 : {}".format(numerical_gradient(function_2, x)))

​

x = np.array([0.3, 0.3])

print("x = [0.3, 0.3] 일 때의 기울기 배열 : {}".format(numerical_gradient(function_2, x)))

​

x = np.array([0.01, 0.01])

print("x = [0.01, 0.01] 일 때의 기울기 배열 : {}".format(numerical_gradient(function_2, x)))

# coding: utf-8

# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3

import numpy as np

import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D

​

​

def _numerical_gradient_no_batch(f, x):

    h = 1e-4 # 0.0001

    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    

    for idx in range(x.size):

        tmp_val = x[idx]

        

        # f(x+h) 계산

        x[idx] = float(tmp_val) + h

        fxh1 = f(x)

        

        # f(x-h) 계산

        x[idx] = tmp_val - h 

        fxh2 = f(x) 

        

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 값 복원

        

    return grad

​

​

def numerical_gradient(f, X):

    if X.ndim == 1:

        return _numerical_gradient_no_batch(f, X)

    else:

        grad = np.zeros_like(X)

        

        for idx, x in enumerate(X):

            grad[idx] = _numerical_gradient_no_batch(f, x)

        

        return grad

​

​

def function_2(x):

    if x.ndim == 1:

        return np.sum(x**2)

    else:

        return np.sum(x**2, axis=1)

​

​

def tangent_line(f, x):

    d = numerical_gradient(f, x)

    print(d)

    y = f(x) - d*x

    return lambda t: d*t + y

     

if __name__ == '__main__':

    x0 = np.arange(-2, 2.5, 0.25)

    x1 = np.arange(-2, 2.5, 0.25)

    X, Y = np.meshgrid(x0, x1)

    

    X = X.flatten()

    Y = Y.flatten()

    

    grad = numerical_gradient(function_2, np.array([X, Y]) )

    

    plt.figure()

    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")

    plt.xlim([-2, 2])

    plt.ylim([-2, 2])

    plt.xlabel('x0')

    plt.ylabel('x1')

    plt.grid()

    plt.draw()

    plt.show()

k : 출력층의 뉴런 개수

강아지, 고양이, 말을 예측 하면 k는 3 - 클래스는 [0, 1, 2]

MNIST 손글씨 데이터 셋이면 k는 10 - 클래스는 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

보통 신경망에서는 MSE를 잘 쓰지 않고 Cross Entropy Error를 활용

MSE는 신경망으로 회귀를 할 때 많이 사용

MSE를 배우는 이유는 말 그대로 loss에 대한 이해를 하기 위함

MSE는 신경망을 우리가 공부 할 때 개념을 익히는 데에 좋다. ( 실무에서는 사용 잘 안한다. )

정상적인 1n을 사용하지 않고 12을 사용한 이유는

MSE를 미분 했을 때 남는게 순수한 오차라고 할 수 있는 (y−t)만 남기 때문에

배치란?

데이터의 묶음

묶음 대로 결과물이 계산 된다.

100개 데이터를 한꺼번에 묶어서(배치를 만들어서) 입력을 했으면, 거기에 대한 결과물도 100개가 한꺼번에 나온다.

배치를 적용한 Loss의 수식은?

N이면 N건에 대한 CEE 값을 구한 다음( 각각이라곤 했지만 한꺼번에 구해진다 )

그 값들을 모두 더하고 N에 대한 평균을 구한다.

배치를 적용한 CEE

CEE=−1N∑n∑ktnklogynk

미니배치란?

MNIST의 데이터의 개수는 60,000건

신경망이 MNIST를 학습 하고, 거기에 대한 평가를 내릴 때 60,000건 모두에 대한 손실 함수의 합을 구해야 할까?

데이터의 양이 굉장히 많은 경우에는 모든 데이터를 다 쓰는 것이 아니고, 데이터의 일부를 랜덤하게 추려서 근사치로 이용할 수 있다.

이 일부가 되는 데이터를 미니배치라고 한다.

미분

한 순간(한 순간은 구할 수가 없으니까 매우 작은 변화에서의) 변화량(기울기)를 구하는 것

변화량이 0 일때를 순간이라고 할텐데, 변화량이 0일때의 기울기는 구할 수가 없기 때문에

변화량 : y의변화량x의변화량

x 한 순간 (x에 대해 매우 작은 변화가 일어났을 때)의 변화량

df(x)dx=limh→0f(x+h)−f(x)(x+h)−x=limh→0f(x+h)−f(x)h

수치미분의 예시

f(x)=y=0.01x2+0.1x

편미분

위의 예에서는 x(인수)가 1개

2개 이상의 인수에 대한 미분을 편미분 이라고 한다.

한 쪽만 미분 하는 것

여러 개의 인수 중 하나만 미분 하는 것

f(x0,x1)=x20+x21

기울기( gradient )

x0에 대한 기울기와, x1의 기울기를 따로 따로 구해서 확인

x0=3,x1=4 일때의 미분을 각각 구함

∂f∂x0, ∂f∂x1 각각 구해봄

각 방향의( x0, x1 ) 기울기를 하나로 묶어서 벡터화 시킨다.

즉 우리는 (∂f∂x0,∂f∂x1)를 구할 것이다.

방향을 알 수 있게 된다.

