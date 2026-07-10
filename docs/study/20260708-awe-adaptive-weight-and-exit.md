# AWE (Adaptive Weight & Exit) — 하나의 '놀람' 신호로 추론 연산을 조절하기

> **스터디 문서 / 학습용 심화 정리.**
> 작성: 유동완 · 2026-07-08
>> 원본 repository : `~/research/awe`
>
>(README · PROJECT.md · docs/RESULTS.md · docs/intuition.md · docs/exp_logs/LOG.md)

---

## 목차

---

##  선행지식

>**학습 vs 추론(test-time)**
>그 밖의 개념들은 본문에서 필요할 때 다시 설명.

### 학습(training) vs 추론(inference) — 두 개의 시간대

| 구분 | 언제 | 무엇을 |
|---|---|---|
| **학습(training)** | 배포 전, 오래 | 데이터로 가중치를 맞춤 |
| **추론(inference / test-time)** | 배포 후, 실시간 | 고정된 모델로 새 문제에 답 |

이 연구의 고안점은 **추론(test-time)** 이다. 보통 추론 때는 모델이 안 바뀐다고 여기지만, **AWE는 "추론 중에도 연산을 더 쓰자(더 생각/자기수정)"** 를 다룬다. 그래서 이름에 **test-time**이 붙는다.

> 그 밖의 핵심 개념은 그때그때 다시 설명한다.

---

## 1. 요약

- **한 문장**: 학습이 끝난 AI가 어려운 문제를 만났을 때 "연산을 더 쓰는" 두 가지 방법 — **1. 더 오래 생각하기**, **2. 그 문제에 맞게 자기를 살짝 고치기** — 을 **'놀란 정도(surprise)'라는 신호 하나로 동시에 조절**할 수 있는지 검증한 연구.
- **이름**: **AWE = Adaptive Weight & Exit** ("적응형 **가중치**(Weight)와 **종료**(Exit)").
- **결론(정직하게)**:
  - ✅ 두 손잡이 각각은 **잘 작동**한다.
  - ❌ 하지만 "**놀람 신호 하나가 두 손잡이를 모두 최적으로 굴린다**"는 주장은 **틀렸다**. 실험으로 반증됨.
  - 💡 밝혀진 것: **"몇 번 더 생각할지"와 "기억을 얼마나 고칠지"는 서로 다른 신호를 봐야 한다.**
- **규모**: 0.2~0.9M 파라미터의 아주 작은 모델 + 합성(가짜) 과제. → **메커니즘을 규명하는 파일럿**, 대형 모델에 대한 증거는 아직 아님.

---

## 2. 배경: '추론 시점(test-time)에 연산을 더 쓴다'는 게 뭔가

딥러닝 모델은 보통 두 단계로 나뉜다.

1. **학습(training)**: 방대한 데이터로 내부 파라미터(가중치)를 조정. 오래 걸리고, 한 번 끝내면 고정.
2. **추론(inference / test-time)**: 학습이 끝난 모델에 새 입력을 넣어 답을 얻는 단계.

보통은 "추론 = 고정된 모델이 한 번 계산하고 답을 뱉는 것"이라고 생각한다. 그런데 **어려운 문제일수록 추론 때 연산을 더 쓰면 정답률이 오른다**는 게 최근 연구의 큰 흐름이다(사람이 어려운 문제에 시간을 더 쓰는 것과 같음). 연산을 더 쓰는 방법은 크게 둘:

| 방법 | 대표 연구 |
|---|---|
| **1. 더 오래 생각하기 (깊이/depth)** | Coconut, Geiping recurrent-depth |
| **2. 그 입력에 맞게 자기를 고치기 (가중치/weight)** | Titans, PonderTTT |

2는 추론 중에 모델이 자기 파라미터를 즉석에서 조금 바꾸는 것 — **"이 문제만을 위한 즉석 수정"**에 가깝고, 이를 **Test-Time Training(TTT)** 이라 부른다.

**기존 연구는 보통 둘 중 하나만** 사용한다. AWE에서의 질문:

> **이 둘을, 'surprise'라는 신호 하나로 동시에 굴릴 수 있는가?**
> - surprise ↑ (아직 이해 못 함) → **더 생각하고** + 기억을 **더 고친다**
> - surprise ↓ (수렴함) → **멈추고** + 기억 갱신도 **멈춘다**

---

## 3. 바탕 개념

### 3-1. latent(잠재) = 모델의 '머릿속' 표현

- **겉(관측)**: 입력 단어, 출력 단어 — 사람이 읽을 수 있음.
- **속(latent)**: 그 사이에서 계산되는 숫자 벡터들 — 모델만 아는 내부 정보들.
- **latent 스텝**: 답을 내기 전에 이 내부 표현을 **한 번 더 굴려 생각하는 한 단계**. 이걸 몇 번 하느냐가 곧 **'깊이(depth)'**.

### 3-2. 연상 메모리

- 토큰(단어)이 하나씩 들어옴 = 문제가 하나씩 주어짐.
- 각 토큰은 두 부분: **key $k$** ("무엇에 대한 질문인가") + **value $v$** ("그 정답은 무엇인가").
- 모델이 답을 떠올림: **예측 = $Wk$**. 그리고 **진짜 정답 $v$**와 비교.
- 이게 **연상 메모리(associative memory)**: 열쇠 $k$를 주면 얽힌 내용 $v$를 떠올리는 저장소.

둘의 차이 $(Wk - v)$ 가 **예측 오차** — 모델이 얼마나 틀렸나.

### 3-3. '틀린 만큼 고쳐 쓰기' = gradient descent (경사하강)

얼마나 못 맞추고 있는지를 숫자 하나로 측정한게 **손실(loss)**:

$$\mathcal{L}(W) = \tfrac{1}{2}\lVert Wk - v\rVert^2 \quad(\text{"정답을 얼마나 못 떠올리나"})$$

이걸 줄이는 표준 방법이 **gradient descent** — 손실의 기울기 반대로 조금 가는 것:

$$W \leftarrow W - \eta\,\underbrace{(Wk - v)\,k^\top}_{\text{gradient}}$$

즉 (내 예측) − (정답) = 오차, 그 오차만큼 모델 가중치를 학습률 $\eta$의 세기로 고쳐 쓴다. **틀린 만큼 손실의 기울기 반대로 가서 기억을 수정.** 원래 이건 *학습 때* 하는 일인데, 어떤 모델 구조는 **추론 중에도 매 토큰마다** 이걸 한다 — 그게 **TTT**.

### 3-4. 핵심 등식: "생각하는 한 걸음 = 학습하는 한 걸음"

fast-weight 계열 모델이 메모리를 갱신하는 실제 공식 = **delta rule**:

$$W_{\text{new}} = W + \beta\,(v - Wk)\,k^\top$$

이걸 3-3의 gradient descent와 나란히 놓으면, $(v-Wk) = -(Wk-v)$ 이므로:

$$W + \beta(v - Wk)k^\top \;=\; W - \beta(Wk - v)k^\top$$

→ **두 식이 완전히 같다.** 즉 모델이 "그냥 기억을 업데이트한다"고 만든 계산이, 뜯어보니 **예측 오차를 줄이는 gradient descent 한 스텝과 똑같다.** 그래서 이 계열을 *"모델이 몰래 학습한다(secretly learn)"* 고 부른다.

> ⚠️ **정확히는 '근사'**: delta rule은 *정확히* GD 1스텝이지만, 실제 Transformer의 softmax attention은 이걸 **비슷하게 흉내** 낼 뿐이다. von Oswald 등(2023)이 선형 attention 한 층의 forward가 회귀 문제의 GD 1스텝을 근사함을 보였다.

### 3-5. surprise의 정체 = gradient의 크기, 그 자체

스터디 중 오해 정정:

- ❌ "surprise가 있어야 → GD가 일어난다" (surprise가 트리거)
- ✅ "GD 식 안의 **오차항 자체가 곧 surprise**다" (같은 것을 두 이름으로)

$$W_{\text{new}} = W - \eta\underbrace{(Wk-v)k^\top}_{\text{gradient}=\text{surprise}}$$

Titans 논문이 실제로 이 gradient를 **surprise로 정의**한다. 그래서:

```
많이 틀림 → gradient 큼 → surprise 큼 → 기억을 크게 갱신
잘 맞음   → gradient 작음 → surprise 작음 → 거의 안 갱신
```

surprise가 0에 가까우면 갱신이 **저절로** 미미해진다 — **on/off 스위치가 필요 없음**. 익숙한 입력엔 자동으로 손을 덜 대고, 낯선 입력엔 자동으로 크게 반응.

### 3-6. → AWE 아이디어: 하나의 신호, 두 개의 손잡이

**surprise는 이미 gradient 안에 공짜로 존재하는 양이다. 그러니 그 하나로 두 가지를 동시에 정할 수 있지 않을까?**

| 질문 | 무엇으로 답하나 | 손잡이 |
|---|---|---|
| 기억을 **얼마나 세게** 고칠까? | 놀란 만큼 | **(W) Weight** — TTT 갱신 |
| **몇 번 더** 생각할까? / 언제 멈출까? | 놀랐으면 더, 익숙하면 그만 | **(E) Exit** — 깊이/halting |

- (W) 갱신은 이미 gradient descent가 **자동으로** 한다.
- (E) halting에 **같은 surprise 신호를 재사용**하자 — 이것이 AWE의 새 제안. **새 신호를 발명하는 게 아니라, GD가 이미 뱉고 있는 surprise를 "언제 그만 생각할지"에도 꽂는 것.**

---

## 4. 메커니즘, 의사코드

```text
s = embed(problem)                 # latent 사고 (내부 표현)
W = W_base                         # 에피소드 fast-weight (질문마다 리셋)
for t in range(max_steps):
    ctx      = lookup(s, context)                 # 한 번의 reasoning "hop"
    hat      = W @ LN(s)                           # 메모리가 그 hop을 예측
    surprise = mean((hat - ctx)^2)                 # ★ 공유 신호 (MSE)
    s        = step(s, ctx, hat)                   # 사고를 한 걸음 전진
    if TTT:  delta = (1-decay)*delta - lr * dSurprise/dW   # [Weight] 갱신
    if EXIT and surprise < tau: break                       # [Exit] 멈춤
answer = readout(s)
```

- **write(쓰기)** 는 정규화(`/‖s‖²`)된 delta-rule + forgetting gate + first-order/detached → 루프 안에서 메모리가 발산하지 않게 안정화.
- **exit(종료)** 는 surprise가 임계값 `tau`보다 낮아지면 멈춤. (구현 팁: 한 번 굴린 surprise 궤적에 사후로 tau를 적용해 하나의 rollout으로 모든 tau를 평가.)

### 4-1. 실제 모델 구성 (아키텍처·하이퍼파라미터)

위 의사코드를 구현한 실제 모델은 **작은 recurrent-depth 추론기 + fast-weight 연상 메모리** 조합이다(파라미터 0.2–0.9M — 합성 과제용 소형).

**세 부분** (`models/recurrent.py`·`ttt.py`):
1. **Prelude(인코더)**: 토큰 임베딩 + 위치 임베딩 → Transformer 인코더(층 2, head 4, FFN 4d, GELU) → LayerNorm. 입력(그래프 표·시작 노드)을 토큰별 벡터 $H$로 인코딩.
2. **Core(한 hop = latent step)**: 현재 latent $s$로 $H$를 cross-attention 조회(`lookup`) → MLP → residual로 $s$ 갱신. 이 한 스텝이 "함수 $f$를 한 번 적용(다음 노드로 이동)"이고, 여러 번 반복 = 여러 hop = 깊이.
3. **Coda(readout)**: LayerNorm → Linear($d$→노드 수)로 답 분포 출력.

**fast-weight 메모리** (공유-신호 모델 `ttt.py`) — *fast-weight* = 학습으로 고정된 보통 가중치와 달리 **추론 중 즉석 갱신되는 가중치**로, key→value를 써넣고 조회하는 **연상 메모리(단기 작업기억 격)**다:

- 에피소드 메모리 $W = W_{\text{base}} + \delta$: **$W_{\text{base}}$**는 학습으로 고정된 기본 메모리, **$\delta$(델타)**는 *이번 입력에만 해당하는 증분* — 질문마다 0에서 시작해 latent step을 돌며 delta-rule로 갱신되고(추론 중 TTT), 다음 질문에선 리셋된다. $\delta$는 **detached**(이 갱신엔 메타-그라디언트를 흘리지 않음 = 1차).
- 매 스텝: $ctx=\text{lookup}(s)$, $\hat{c}=W\,\text{LN}(s)$, $surprise=\lVert\hat{c}-ctx\rVert^2$, $s\leftarrow\text{step}(s,ctx,\hat{c})$; [쓰기] $\delta$를 delta-rule로 갱신(정규화·error-gated·forgetting), [멈춤] $surprise<\tau$면 종료.
- 양성 결과 모델(`memory.py`)은 쓰기는 같은 정규화 delta-rule(lr 0.5, decay 0.1)이되 **멈춤은 readout 엔트로피**로 한다.

**학습·설정** (정본 `reachp2`):
- 은닉 차원 $d=256$(기본 128), head 4, 인코더 층 2, FFN 4d, max_len 64.
- 학습: 8000 step, batch 64, lr 3e-4, next-node **보조 손실**(가중치 1.0).
- **K-커리큘럼**: 난이도 kcap을 학습 중 1→4로 램프, 평가는 kcap=4.
- latent step 예산 $T$: `reachp2`=6, 깊이 sanity(Part 1)=12. seeds 10.

---

## 5. 실험 설계: '사다리' 전략

**작은 합성 과제** 위에서, 손잡이를 **하나씩 격리**해 검증한 뒤 합치고, 마지막에 외부 과제로 전이되는지 본다.

### 5-1. 검증 사다리

| 단계 | 실험 | 격리한 것 | 답하려는 질문 |
|---|---|---|---|
| **Part 1** | in-context 도달성(reachability) | 깊이(E)만 | 더 깊이 생각하면 더 어려운 문제를 푸나? 멈춤이 난이도를 따라가나? |
| **Part 2** | 숨은 규칙(hidden-rule) | 가중치(W)만 | 추론 중 기억을 쌓으면 정답률이 실제로 오르나? |
| **Part 3** | 부분관측 도달성(joint reachp) | 둘 다 | 한 신호로 깊이+가중치를 **동시에** 굴리면 잘 되나? |
| **Part 4** | halting 신호 **bake-off** | 신호 비교 | Part 3이 실패했다면, **어떤 신호**가 halting에 맞나? |
| **Part 5** | MQAR (외부 과제) | 전이 | Part 4의 결론이 **다른 과제로도** 전이되나? |

**과제 직관**:
- **도달성(reachability)**: 함수 그래프에서 "여기서 K번 따라가면 어디 도착?" — **K(hop 수) = 난이도**. K가 크면 latent 스텝이 더 필요 → 깊이 손잡이 시험.
- **숨은 규칙(hidden-rule)**: 감춰진 순열 π를 질문 스트림에 조금씩만 흘림. 각 질문은 이전 질문들에서 **쌓인 기억으로만** 풀 수 있음 → 기억 손잡이 시험.
- **부분관측 도달성(reachp/reachp2)**: 위 둘을 합침. `reachp2`는 base 모델 강화판(K-커리큘럼 + 보조손실 + 폭 d=256).
- **MQAR** (Multi-query Associative Recall): 앞서 '키→값' 쌍 여러 개를 흘려준 뒤 특정 키를 다시 물으면 그에 짝지어진 값을 기억해 답하는 **연상 회상** 과제. associative recall 계열의 외부 벤치마크(Zoology/Based에서 쓰이는 (준)표준). **본 연구는 축소판(mini, vocab 64) 위주 + 표준 config(vocab 8192)는 앵커로 1회 확인.** 우리 합성 과제 밖에서 결론이 서는지 보는 외부 시험대(single-hop / multi-hop).

### 5-2. 4-way 비교 (ablation)

| 구성 | 깊이 halting | 기억 지속 | 뜻 |
|---|:--:|:--:|---|
| **fixed** | ✗ | ✗ | 아무것도 안 함 (바닥선) |
| **+halt** | ✓ | ✗ | 깊이 손잡이만 |
| **persist** (+ttt) | ✗ | ✓ | 기억 손잡이만 |
| **both (AWE)** | ✓ | ✓ | 둘 다 (제안) |

### 5-3. "언제 멈출지"를 정하는 신호 (Part 4의 핵심)

halting은 "latent 스텝을 언제 그만 밟을지"를 정하는 일이다. 방식은 단순하다: **매 스텝마다 어떤 수치(신호)를 재고, 그 값이 임계값 `tau`보다 낮아지면 '다 됐다'고 보고 멈춘다.** 진짜 문제는 **어떤 수치를 재느냐** — 후보를 세 계열로 나눠 같은 궤적 위에서 공정하게 경쟁시켰다.

**① 확신도 계열 — `entropy` (기존 관행)**
지금 답 분포가 얼마나 헷갈리는지. 엔트로피가 낮으면 "이미 확신함 → 그만". 대부분의 기존 halting이 쓰는 신호.

**② 재구성 오차 계열 — `recon` (우리 가설이 지목한 신호)**
메모리가 다음 hop을 얼마나 **못 맞췄는지** = §3-5의 그 **surprise** 그 자체. 오차가 낮으면 "메모리가 잘 맞음 → 그만". *가설대로라면 이게 halting에서 이겨야 한다.*

**③ 수렴 계열 — `conv`, `dstate`**
"내부 계산이 더 이상 **안 움직이는가**"를 봄 (Geiping식 "안 움직이면 멈춰라").
- `conv`: **예측(readout)**이 스텝 사이 얼마나 안 변하나 (연속 두 스텝 분포의 sym-KL).
- `dstate`: **내부 상태**가 스텝 사이 얼마나 안 변하나 ‖Δs‖². 이론상 ≈ ‖∇ₛL‖(= surprise의 gradient 크기)라, §3–5 이론이 실제로 가리키는 신호.

(그 외 보조 후보: `rnorm`=읽기 벡터 크기, `dent`=엔트로피 변화량.)

| 신호 | 무엇을 재나 | 값이 낮으면 | 이론상 정체 |
|---|---|---|---|
| `entropy` | 답 분포의 불확실성 | 확신함 → 멈춤 | readout 확신도 |
| `recon` | 메모리의 다음-hop 예측 오차 | 메모리가 맞음 → 멈춤 | **문자 그대로의 surprise** |
| `dstate` | 내부 상태 변화량 ‖Δs‖² | 상태가 수렴 → 멈춤 | ≈ ‖∇ₛL‖ (surprise gradient) |
| `conv` | 예측 변화량 (sym-KL) | 예측이 수렴 → 멈춤 | 수렴 신호 (Geiping식) |

> 우리 가설이 예측하는 신호는 `recon`(과 그 gradient인 `dstate`)이다. **과연 `recon`이 halting에서 이기는가**가 실험의 진짜 쟁점 — 결과(§7 Part 4)는 "아니다"였고, 대신 **수렴 계열**이 이겼다.

### 5-4. 숫자를 믿게 만드는 장치 (공정성 규율)

- **10 seeds**: 모든 결과는 서로 다른 난수 10회의 평균±표준편차.
- **held-out tau**: 멈춤 임계값 τ를 **평가와 다른 배치**에서 보정 (test에서 유리하게 고르는 부정 방지).
- **shuffle null (헛것 대조군)**: 신호를 블록 안에서 뒤섞어도 성능이 같으면 그건 진짜 "예제별 정보"가 아님. 이걸 이겨야 유의미.
- **tau 선택 목적 = fewest-steps-within-slack**: "정확도 손해 ≤1pp 안에서 가장 적은 스텝"을 고름. (초기엔 argmax-accuracy를 썼는데, 이게 Part 4a의 잡음 원인 → Part 4c에서 교정.)

---

## 6. 결과를 읽기 전에 — 이야기의 3막극

전체 스토리는 **"실패 → 진단 → 반전 → 전이"** 다. Part 3에서 순진한 버전이 깨지고(1막), Part 4에서 "메커니즘이 아니라 신호 선택의 문제"임을 진단해 반전시키고(2막), Part 5에서 그 결론이 외부 과제로도 전이됨을 확인(3막)한다.

---

## 7. 실험 결과 (모든 수치 10 seeds 평균; 원본: AWE 레포 `docs/RESULTS.md`)

### Part 1 — 깊이만: 잘 된다 ✅

더 깊이 생각할수록 정확도가 오르고(r=1: **15.7%** → r=12: **100%**), 멈추는 시점이 난이도 K를 거의 완벽히 따라감:

$$\text{corr}(K,\ \text{halt step}) = +0.997 \pm 0.001 \quad(\text{per-K conv-step} = K \text{ 정확 일치})$$

→ 과제가 "깊이 손잡이"를 시험할 올바른 난이도 구조를 가짐. (여기 halting은 사후 수렴 판정이라 surprise를 직접 시험한 건 아직 아님.)

### Part 2 — 기억만: 가장 강한 결과 ✅

기억을 **지속(persist)** 하면 스트림을 따라 정답률이 **5% → 81%** 로 상승. 리셋하면 계속 찍기(~5%)에 머묾 → 기억이 **필수**.

| 구성 | 정확도 | 평균 latent 스텝 |
|---|---|---|
| fixed / +halt (리셋) | ~5% | — |
| **persist** | 50.1% | 8.0 |
| **both** | 50.0% | **2.4** |

→ `both`는 같은 정확도를 **8.0 대신 2.4 스텝**(≈3.3× 적은 조회)으로 달성. 스트림을 따라 스텝이 **7.95 → 1.21** 로 줄어드는 amortization 확인. `corr(정답가능, entropy) = −0.96` (잘 보정된 신호).

### Part 3 — 둘 다 합치니: 실패 🟡 (그러나 원인이 있다)

**halting을 켜면 정확도가 깎임**:

| 구성 | 정확도 | 평균 스텝 |
|---|---|---|
| fixed | 21.0% | 10.0 |
| +halt | 16.0% | 3.2 |
| **persist** | **35.3%** | 10.0 |
| **both (AWE)** | 25.7% | 5.4 |

- `both`(25.7%)가 `persist`(35.3%)보다 **−9.6pp**. 일찍 멈추긴 하는데 **확신에 차서 틀린 채(confident-but-wrong) 조기 종료**하는 예제가 많음.
- 당시 남은 질문: 이게 **메커니즘의 실패인가, 신호 선택의 실패인가?**

### Part 4 — 진단: '신호 선택' 문제였다

base 모델을 강화(reachp2: 커리큘럼+보조손실+d=256)해 천장을 **72.0%** 로 올린 뒤, **같은 궤적 위에서 여러 halting 신호를 맞대결**.

**Part 4a (강한 학습기, 10 seeds):**

| halt 신호 | 정확도 | 평균 스텝 | persist 대비 | 조기-오답 |
|---|---|---|---|---|
| `dstate` (‖Δs‖²) | **72.0%** | 5.93 | **−0.0pp** | **1.1%** |
| `conv` (sym-KL) | 71.9% | 5.00 | **−0.0pp** | 3.6% |
| `entropy` (기존) | 66.4% | 5.44 | −5.6pp | 8.5% |
| `recon` (재구성 오차) | 66.3% | 5.39 | −5.6pp | 8.8% |

→ **−5.6pp의 손해는 신호 특유의 문제.** `entropy`·`recon`은 공격적으로 멈추되 잘못 보정돼 ~9%가 "확신하고 틀린 채" 조기 종료. **수렴 계열(`conv`, `dstate`)은 천장 정확도를 그대로 유지.**

**Part 4c (정본/canonical, 10 seeds, 엄격 프로토콜):**

| 신호 | 정확도 | 스텝 | slack 내 작동점 존재? |
|---|---|---|:--:|
| **`conv`** | **71.2%** | **3.89±0.09** | ✅ **10/10** |
| `dstate` | 71.0% | 5.12 | ✅ 10/10 |
| `dent` | 68.8% | 5.70 | ❌ 0/10 |
| `rnorm` | 68.2% | 5.57 | ❌ 0/10 |
| `recon` | 67.1% | 5.45 | ❌ 0/10 |
| `entropy` | 66.2% | 5.43 | ❌ 0/10 |

- **`conv`는 ~35% 연산을 절약**(3.89/6 스텝)하며 정확도 손해 ≤1pp, **10/10 seed 균일**. Part 4a의 bimodal/보수성은 τ 규칙의 잡음이었고 여기서 해소.
- **구별이 이분법**이 됨: slack(≤1pp) 안에서 작동점을 찾는 신호는 **수렴 계열(`conv`/`dstate`)뿐**(10/10). 나머지 넷은 0/10.
- `conv`는 shuffle null을 **+7.4pp** 이김 → 진짜 예제별 정보를 담음.

### Part 5 — 외부 전이

외부 벤치마크 **MQAR**(associative recall 계열)로도 재현되는가?

- **Multi-hop MQAR** (hop 수 = 난이도): **수렴 계열이 entropy/recon을 +2.5pp, 10/10 seed로 이김** (2.5 vs 5.4 스텝, 절반 이하 연산). 깊이가 hop 수 따라 증가(2.75→3.51). → reachp의 결론이 **외부 과제로 일반화**.
- **Single-hop MQAR**: `conv`가 천장을 **2.53/6 스텝(~58% 절약)** 으로 전이하지만, 이 과제는 신호를 **구별하지 못함**(오답이 애초에 못 푸는 문제라 조기 종료가 무해). → 전이는 되나 구별력은 multi-hop에서만 드러남.

---

## 8. 종합 결과

- ✅ **깊이(E) 손잡이 성립** — 단, 올바른 신호는 **수렴 계열(`conv`·`dstate`)**(그중 연산 효율 대표는 `conv`). "더 생각할지"는 *예측이 더 안 변하는가*로 판단해야 잘 된다.
- ✅ **가중치(W) 손잡이 성립** — 재구성 오차로 게이팅한 delta-rule 갱신이 잘 작동(정규화·decay·작은 inner-lr로 안정화).
- ❌ **주 가설은 실패** — "**하나의 재구성 오차 스칼라(`recon`)가 두 손잡이를 모두 최적 구동**"은 성립 안 함. `recon`은 모든 joint 과제에서 halting에 **패배**. 연산을 아끼는 halter(`conv`)는 **읽기(readout) 통계**라 write 역할이 없음.

> **핵심 결론: 깊이와 가중치는 서로 다른 관측량을 원한다.**
> 깊이 ← 수렴(`conv`·`dstate`), 가중치 ← 재구성-미스(`recon`).

§3의 이론("latent 스텝 = GD, surprise = gradient")은 **부분적으로 살아있음.** 그 이론이 가리키는 신호 `dstate`(≈‖∇ₛL‖)는 정확도를 **보존**(−0.0pp)하지만, *가장 연산 효율적인* halter는 아니었음(보수적으로 멈춤).

즉 **"두 손잡이 = 하나의 메모리 손실의 두 gradient(∇ₛL / ∇_WL)"** 라는 약한 해석은 **모순은 아니나 아직 미검증**. "발견"이 되려면 **halt 신호가 write 크기를 실제로 예측함**을 보여야 함(다음 실험: write-magnitude probe).

- "연산 절약"은 **latent 조회 스텝만** 셈 (delta-rule write FLOPs는 halting과 무관, 별도 회계 필요).
- 모델은 **0.2–0.9M 파라미터 + 합성 과제** — 메커니즘 파일럿이지 LLM 규모 증거 아님.

---

## 9. 지금까지의 결과와 다음 (2026-07-07 기준)

**완료(✅)**: 깊이-only 증명 · 메모리-only 증명 · joint 실패의 원인 진단 · halting-신호 bake-off ×2(독립 재현) · 정본 bake-off · 외부(MQAR) 전이 · held-out tau/다시드/인프라 정비.

**다음(codex + critic 교차리뷰가 수렴한 방향)**:
1. **targeted backfill**: ① Part 1 `depth_sanity` 다시드+로그 보존은 **완료(2026-07-07, 10 seeds, corr +0.997±0.001)** ② 표준 config(vocab 8192) zoology-MQAR 앵커 — single-hop 완료, **multi-hop은 미실행**(공개 baseline Based/DeltaNet 대비). *mini-MQAR base learner 튜닝은 목적 아님.*
3. **다음 실험**: write-magnitude 질문을 **ICL-회귀 개입실험**으로 재설계(반사실적 write 조작 + 함수형 예측 + cross-regime 불변성). 지금의 passive 상관은 이중 confound라 폐기.

---

## 10. 용어

| 용어 | 뜻 (한 줄) |
|---|---|
| **latent / hidden** | 입력·출력 사이, 모델 내부의 숫자 벡터 표현 (사람이 못 읽음) |
| **latent 스텝** | 답을 내기 전 내부 표현을 한 번 더 굴려 생각하는 한 단계 (= 깊이) |
| **연상 메모리** | 열쇠 $k$를 주면 얽힌 내용 $v$를 떠올리는 저장소 $W$ |
| **key $k$ / value $v$** | 토큰의 "질문" 부분 / "정답" 부분 |
| **손실(loss)** | 예측이 정답에서 얼마나 벗어났나를 재는 숫자 |
| **gradient descent** | 손실의 기울기 반대로 조금 가서 손실을 줄이는 학습 스텝 |
| **delta rule** | fast-weight 메모리 갱신 공식; GD 1스텝과 동일 |
| **surprise** | 그 gradient의 크기 = 모델이 얼마나 놀랐나 (= 얼마나 틀렸나) |
| **TTT (Test-Time Training)** | 추론 중에 입력에 맞춰 가중치를 즉석 갱신 |
| **halting / Exit** | latent 스텝을 언제 그만 밟을지 결정 (= 깊이 조절) |
| **ablation** | 손잡이를 켜고/끄며 기여를 분리하는 비교 (fixed·+halt·persist·both) |
| **seed** | 난수 시드. 10 seeds = 서로 다른 초기화 10회로 평균±표준편차 |
| **tau (τ)** | halting 임계값. 신호가 이보다 낮아지면 멈춤. held-out에서 보정 |
| **`conv`** | 예측이 스텝 사이 안 변하는 정도(수렴). **깊이의 올바른 신호로 판명** |
| **`dstate`** | latent 상태 변화량 ‖Δs‖² ≈ ‖∇ₛL‖. 이론이 가리키는 surprise gradient |
| **`recon`** | 재구성 오차 = 문자 그대로의 surprise. write엔 맞지만 halting엔 패배 |
| **MQAR** | Multi-query Associative Recall. associative recall 계열의 (준)표준 외부 벤치마크(Zoology/Based) |
| **pp** | percentage point. 정확도 차이 단위 (예: −5.6pp) |
| **amortization** | 기억이 쌓일수록 문제당 필요한 스텝이 줄어드는 현상 |

---

## 11. 원본 문서 구성도

AWE 레포는 계층적으로 문서화되어 있어, 필요한 깊이에 따라 골라 읽으면 된다:

| 문서 | 역할 | 언제 보나 |
|---|---|---|
| `README.md` | quickstart + 상태 요약 | 처음/빠르게 |
| `docs/intuition.md` | **한국어 from-zero 직관 설명** | 개념이 낯설 때 (이 스터디 문서의 원천) |
| `PROJECT.md` | 전체 서사 · 가설 · 로드맵 | 맥락 전체를 볼 때 |
| `docs/RESULTS.md` | Part 1–5 정량 결과 + 신호 인벤토리 | 수치를 인용할 때 (정본) |
| `docs/mqar_design.md` | 외부 과제 설계 + MQAR 결과 | Part 5 상세 |
| `docs/proposal.md` | 연구 proposal + 문헌 위치 | 관련 연구 대비 |
| `docs/exp_logs/LOG.md` | 날짜별 실험 인덱스 (판정 태그 ✅🟡🔴) | 실험 이력 추적 |
| `LOGGING.md` | 실험 기록 규약 (run log → results/, 인덱스 한 줄) | 재현·기록할 때 |

---

## 12. 참고 문헌

**'forward 계산이 곧 학습'**
- Schlag et al. 2021, *Linear Transformers Are Secretly Fast Weight Programmers* — delta rule = fast weight = 숨은 GD.
- von Oswald et al. 2023, *Transformers Learn In-Context by Gradient Descent* — attention 한 층의 forward가 회귀 GD 1스텝을 근사.
- Sun et al. 2024, *TTT layers* (2407.04620) — hidden state가 test-time에 self-supervised loss로 학습.

**surprise / 메모리 갱신**
- Behrouz et al. 2024, *Titans* (2501.00663) — gradient를 surprise로 정의, momentum + forgetting.
- *PonderTTT* (2601.00894, 2026) — TTT 재구성 손실로 weight 갱신을 이진 게이팅.

**latent 깊이 / halting**
- *Coconut* (2412.06769) — latent 추론(스텝 수 고정).
- Geiping et al. 2025, *recurrent-depth* (2502.05171) — latent 깊이 스케일링, per-token 종료를 KL-수렴으로.
- *FR-Ponder* (2509.24238) — 적응형 latent 깊이 halting을 RL(GRPO)로.

**같은 실험 niche (인용·차별화 대상)**
- *UT-Memory* (2604.21999) — 깊이–메모리 trade-off(단, train-time 용량 + 학습된 ACT 라우터).
- *HRM* (2506.21734) / *TRM* (2510.04871) — 작은 재귀 모델 + 학습된 Q-halting.

---

*Research WIP — Dongwan Yoo, 2026. 정량 결과·재현 방법은 AWE 레포의 원본 문서를 정본으로 삼는다.*
