# RL_Project
# Domain Randomization이 물리적 환경 변화에 대한 강화학습 기반 로봇 제어의 강건성에 미치는 영향

**The Effect of Domain Randomization on the Robustness of Reinforcement Learning-Based Robotic Control under Physical Parameter Variations**

## 1. 프로젝트 소개

본 프로젝트는 MuJoCo 물리 시뮬레이션 환경에서 강화학습 기반 로봇 팔 제어 모델을 구현하고, Domain Randomization 적용 여부에 따른 로봇 제어의 강건성과 일반화 성능을 비교·분석하는 것을 목표로 한다.

일반적인 강화학습 모델은 특정한 물리적 환경에서 학습하기 때문에, 학습 당시와 다른 물리적 조건이 주어지면 제어 성능이 저하될 수 있다.

이를 해결하기 위한 방법 중 하나인 \*\*Domain Randomization(DR)\*\*은 학습 과정에서 환경의 물리적 파라미터를 무작위로 변경하여 다양한 조건을 경험하도록 하는 기법이다.

본 프로젝트에서는 로봇 팔의 \*\*질량(Mass)\*\*과 \*\*관절 감쇠 계수(Joint Damping)\*\*를 주요 변경 변수로 설정한다. 이를 통해 각 물리적 요소의 개별적인 영향과 복합적인 변화가 로봇 제어 성능에 미치는 영향을 분석한다.

또한 학습 과정에서 경험하지 못한 물리적 조건에서도 안정적으로 목표에 도달할 수 있는지 평가하여 Domain Randomization의 일반화 효과를 확인한다.

## 2. 연구 목표 및 질문

### 연구 목표

- MuJoCo 기반 2관절 로봇 팔 제어 환경 구현
- PPO(Proximal Policy Optimization)를 활용한 로봇 제어 모델 학습
- 질량 및 관절 감쇠 계수의 무작위화에 따른 성능 변화 분석
- 학습 범위 내·외의 물리적 환경에서 강건성 및 일반화 성능 평가

### Research Questions

- **RQ1.** Domain Randomization을 적용한 모델은 고정된 환경에서 학습한 모델보다 물리적 환경 변화에 강건한가?
- **RQ2.** 질량과 관절 감쇠 중 어떤 물리적 파라미터를 무작위화하는 것이 제어 성능 향상에 더 효과적인가?
- **RQ3.** 질량과 관절 감쇠를 동시에 무작위화하면 개별적으로 무작위화했을 때보다 더 높은 강건성을 보이는가?
- **RQ4.** 학습 과정에서 경험하지 못한 물리적 조건에서도 성능을 유지할 수 있는가?

## 3. 연구 환경 및 기술 스택

| 구분                     | 내용                                           |
| ---------------------- | -------------------------------------------- |
| Physics Simulator      | MuJoCo                                       |
| Environment            | Gymnasium MuJoCo Reacher                     |
| Reinforcement Learning | PPO                                          |
| Algorithm Library      | Stable-Baselines3 (예정)                       |
| Programming Language   | Python                                       |
| Main Variables         | Mass, Joint Damping                          |
| Task                   | Robotic Arm Reaching                         |
| Evaluation             | Success Rate, Average Return, Final Distance |

### 로봇 제어 과제

2관절 로봇 팔의 끝부분(End-effector)이 주어진 목표 위치에 도달하도록 학습한다.

- **State:** 관절 위치, 관절 속도, 목표 위치 등 환경에서 제공하는 관측값
- **Action:** 두 관절에 적용하는 연속형 토크 제어
- **Reward:** 목표와의 거리 및 환경의 기본 보상 구조 활용
- **Policy:** PPO 기반 제어 정책

## 4. 실험 설계

### 4.1 학습 환경 구성

총 4개의 실험군을 구성하여 Domain Randomization 적용 방식에 따른 성능을 비교한다.

| 실험군 | 질량(Mass)     | 관절 감쇠(Joint Damping) | 설명              |
| --- | ------------ | -------------------- | --------------- |
| A   | 1.0 고정       | 1.0 고정               | Baseline        |
| B   | 0.8\~1.2 무작위 | 1.0 고정               | 질량만 무작위화        |
| C   | 1.0 고정       | 0.5\~1.5 무작위         | 감쇠만 무작위화        |
| D   | 0.8\~1.2 무작위 | 0.5\~1.5 무작위         | 질량 및 감쇠 동시 무작위화 |

※ 파라미터는 기본 환경의 값을 1.0으로 정규화한 배율이다.

※ Domain Randomization 적용 시 각 에피소드 시작 시 파라미터를 샘플링하고, 에피소드가 진행되는 동안에는 해당 값을 유지한다.

### 4.2 테스트 환경 구성

학습된 모델을 다양한 물리적 조건에서 평가한다.

| 테스트 환경 | 질량 배율 | 감쇠 배율 | 평가 목적           |
| ------ | ----: | ----: | --------------- |
| T1     |   1.0 |   1.0 | 기본 환경           |
| T2     |  1.15 |   1.0 | 질량 변화           |
| T3     |   1.0 |   1.4 | 감쇠 변화           |
| T4     |  1.15 |   1.4 | 복합 변화           |
| T5     |   1.3 |  1.75 | 학습 범위 밖 환경(OOD) |

- T1\~T4: 학습 범위 내 물리적 조건에서의 성능 평가
- T5: 학습 범위를 벗어난 물리적 조건에서의 일반화 성능 평가

### 4.3 평가 지표

| 평가 지표                   | 설명                                 |
| ----------------------- | ---------------------------------- |
| Success Rate            | 전체 에피소드 중 목표 도달에 성공한 비율            |
| Average Return          | 에피소드별 평균 누적 보상                     |
| Final Distance          | 에피소드 종료 시점의 목표와 End-effector 사이 거리 |
| Performance Degradation | 기본 환경 대비 테스트 환경에서의 성능 저하율          |

성능 저하율은 다음과 같이 계산한다.

[
Degradation(%) =
\frac{SR\_{nominal}-SR\_{test}}{SR\_{nominal}}
\times 100
]

단, 기본 환경의 성공률이 매우 낮은 경우 비율만으로 해석하지 않고 성공률 차이(%p)도 함께 확인한다.

각 실험은 동일한 학습 설정과 평가 조건을 유지하며, 여러 Random Seed를 사용하여 결과의 평균과 표준편차를 비교한다.

## 5. 기대 효과

- 물리적 파라미터 변화가 강화학습 기반 로봇 제어에 미치는 영향 분석
- 질량 및 관절 감쇠의 개별·복합 무작위화 효과 비교
- Domain Ran?domization의 강건성 및 일반화 성능 평가
- 기본 환경에서의 제어 성능과 새로운 환경에서의 일반화 성능 사이의 Trade-off 분석


## 7. 팀 구성

| 역할 | 이름  | 
| --- | ---- |
| 팀장 | 박가윤 |
| 팀원 | 김태희 |
| 팀원 | 송예원 |

## 8. References

- Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*. IEEE International Conference on Robotics and Automation (ICRA).
- MuJoCo Documentation. [https://mujoco.readthedocs.io/](https://mujoco.readthedocs.io/)
- Gymnasium Documentation. [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- Stable-Baselines3 Documentation. [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)