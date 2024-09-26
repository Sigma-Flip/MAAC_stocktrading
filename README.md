## ✅ MAAC(Multi-Agent Actor-Critic)을 활용한 주식시장 강화학습 에이전트 생성



### 🟦 Purpose
MARL 알고리즘 중 **MAAC**를 활용하여 효과적인 투자 기법을 학습하는 것을 목표로 합니다. 또한 한국투자증권 API를 활용하여 실질적인 투자에도 적용할 수 있도록 설계하였습니다.

---

### 🟦 Background
기존에도 강화학습을 주식투자에 접목하려는 시도는 많았습니다. 하지만 대부분 몇 가지 한계점들을 가지고 있습니다.

- **offline learning** : 과거 데이터만을 사용해 실시간 학습이 불가능했습니다 
- **Observation Space** : OHLCV 데이터로만 한정되어 있어 하루에 1개의 데이터만을 사용할 수 있어 실질적 활용에 제한이 있습니다.
- **Action Space** : "100% 매수", "유지", "100% 매도" 등으로 설정되어 실질적 투자에 활용하기 어렵습니다.
- **Single Stock Agent RL** : 특정 주식에 대한 최적화만 이루어져 포트폴리오 관리를 위한 일반적인 조건과 거리가 있습니다.
- **dimension exploding** : 단일 에이전트 강화학습은 특정 주식에 대한 최적화만 이루어져 실제 포트폴리오 관리와는 거리가 있었습니다..
- **활용성** : 실제 투자 환경이 아닌 시뮬레이션과 백테스팅에만 의존하는 경우가 많습니다.


---
### 🟦 Key Improvements and Features
- **Online Learning**: online learning을 통해 실시간 학습이 가능하도록 하였습니다.

- **Observation Space**: OHLCV뿐만 아니라, 현재가, PER, PBR, 누적 거래량, 외국인 보유 수량 등 30개 이상의 차원으로 확장하였습니다.
-  **Action Space** :  Discrete Action Space로 설정하고, MAAC를 활용하여 차원이 선형적으로 증가하도록 만들었습니다.
- 한국투자증권 API(mojito)를 사용하여 실제 투자에 활용할 수 있도록 구현하였습니다.
---
### 🟨 Core Sturcture 
- `Central Critic`  : Multihead attention 활용, actions에 대한 values 계산
- `Discrete Policy`   : 각 agents의 action 출력
- `target critic, target policies`   : critic 과 policies 의 안정적 학습 
- `Replay Buffer`   : *S, A, R, S_* 를 저장, 추후에 모델 학습에 활용

---

### 🟨 Stock Market Environment
MARL에서의 환경(Environment)은 크게 세 가지로 나눌 수 있습니다.

1. **Cooperative Environment**: 에이전트들이 협력하여 최적의 행동을 학습합니다 (Maximize long-term return).
2. **Competitive Environment**: 에이전트들이 상대를 이기기 위한 최적의 행동을 학습합니다.
3. **Mixed Environment**: 에이전트들이 팀으로 협동하여 다른 에이전트들과 경쟁합니다.

**주식 포트폴리오 시장**은 **competitive-cooperative environment**에 해당합니다. 기존의 Mixed Environment에서는 팀 간의 협동이 이루어지지만, 주식시장은 서로 다른 양상을 보입니다. 각각의 주식 에이전트는 하나의 공유 자원(초기 자본금)을 경쟁적으로 사용해야 합니다. 

*즉, Agent 1이 주식을 매수하면 Agent 2는 해당 주식을 매수하지 못할 수 있습니다. 이러한 이유로 주식시장은 기존의 Mixed Environment와는 다르게 접근할 필요가 있습니다.*

---

### 🟨 Reward
많은 MARL, 특히 Cooperative RL의 문제는 보상 측정입니다. 여러 에이전트의 행동에 대해 하나의 공통 보상을 설정하는 경우가 많습니다. 보통 **Joint Reward**를 활용해 Advantage Reward를 측정합니다. 하지만 주식시장은 각 주식이 개별적인 보상을 가지므로, Competitive-Cooperative 환경에서는 **Individual Reward**가 적합합니다. 코드에서는 이를 **Sub Reward**로 정의하고 그대로 활용합니다.

---

### 🟨 Multi-Head Attention
MARL의 핵심 과제 중 하나는 에이전트 간의 상호작용입니다. 축구에서 팀 플레이로 골을 넣듯이, 에이전트들끼리도 Observation이나 Action을 공유할 필요가 있습니다. 본 코드는 **Central Critic**에 **Multi-Head Attention**을 활용하여 이를 구현하였습니다.

---


###  과거 프로젝트 
- [연속적 행동 기반 강화학습 주식 에이전트 ](https://github.com/Sigma-Flip/StockTrading_SoftActorCritic_KR_EN)
  
---

### 🔔 Future Improvements
- **Parallel Environment**: 기존 MAAC 논문에서는 병렬 환경에서 학습을 진행합니다. 하지만 이 코드는 실제 증권사 계정과 연결해야 하므로 병렬 학습을 배제하였습니다. 그러나 모의 투자의 경우 계좌를 추가 개설할 수 있음을 확인하였으며, 이는 추후 업데이트될 예정입니다.
- **Reward Function Update**: 현재는 각 주식의 (평가 금액 - 매입 금액)으로 보상을 산정하고 있습니다. 하지만 전체 포트폴리오 보상(Joint Reward)은 고려되지 않는 상태입니다. 이를 활용하여 Advantage Reward Function을 구축할 예정입니다.
- **Model-Based RL**: 현재 MAAC는 실제 주식 환경과 상호작용하여 보상을 얻습니다. 추후에는 환경 모델을 만들어 학습시키는 **Model-Based RL**로 발전시킬 계획입니다.

---

### 🔔 Notifications
- ### 현재 **mojito** 모듈에서 Key Error가 발생하여 코드가 실행이 되지 않는 문제가 있습니다!!! 이는 mojito 자체의 오류입니다.
- ### 코드를 실행하기 위해서는 한국투자증권 회원가입, 계좌 개설 및 Key 발급이 필요합니다!!!
- ### 현재 코드를 실행하기 위한 주식코드 리스트는 직접 작성해야 합니다. config.Stockconfig.stock_codes 에서 수정가능합니다.

---

### References
- [GitHub: MAAC Implementation](https://github.com/shariqiqbal2810/MAAC/tree/master/utils)
- [MAAC 논문](https://arxiv.org/pdf/1810.02912)
- [Stable Baselines Documentation](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html)
