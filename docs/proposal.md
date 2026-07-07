# 연구 제안서 — Surprise-Unified Test-Time Adaptation

> **한 줄 요약**: 하나의 *surprise* 신호가 test-time에 **얼마나 더 생각할지(latent 깊이 halting)** 와 **가중치를 얼마나 바꿀지(fast-weight 갱신)** 를 동시에 조종하도록 통합하고, 그 통합이 각 손잡이를 따로 쓸 때보다 낫다는 것을 실증한다.

- **작성**: 유동완 (DAIS @KIER) · 2026-07-03
- **상태**: 문헌 검증 완료(deep-research 108 agents + 4편 원문 3-pass), 방향 P1 확정
- **키워드**: test-time compute, latent reasoning, test-time training, adaptive halting, surprise

---

## 1. 문제의식 (Motivation)

추론형 LLM은 test-time에 계산량을 늘려 성능을 얻는다. 그 계산을 늘리는 방법은 두 갈래로 나뉜다.

- **(A) 깊이**: 더 오래 생각한다 — latent 추론 스텝을 더 밟는다 *(Coconut, PonderNet, 재귀 깊이)*.
- **(B) 가중치**: 이 입력에 맞게 파라미터를 잠깐 바꾼다 — test-time training *(TTT, Titans)*.

이 두 손잡이는 지금까지 **따로** 연구됐다. 그러나 실제로는 하나의 질문의 두 얼굴이다: *"이 입력이 얼마나 낯선가(surprise)?"* 가 크면 **더 생각해야 하고 + 가중치도 바꿔야 한다.** 작으면(수렴) **멈추고 + 그대로 둔다.** 아무도 이 둘을 하나의 신호로 묶지 않았다.

## 2. 선행연구와 빈 칸 (Gap)

4개 축(A 깊이 · B 가중치 · C surprise 신호 · **D 통합**)으로 최근 논문을 분해하면:

| 논문 | A 깊이 | B 가중치 | C surprise | D 통합 |
|---|:--:|:--:|:--:|:--:|
| FR-Ponder (2509.24238) | ✅ RL(GRPO) | ✗ | ✗ | ✗ |
| Geiping recurrent-depth (2502.05171) | ✅ KL-수렴 | ✗ | ✗(반대) | ✗ |
| PonderTTT (2601.00894) | ✗ | ✅ 이진 게이트 | ~ recon. loss | ✗ |
| Titans (2501.00663) | ✗ | ✅ 메모리 | ✅ surprise→메모리 | ✗ |
| **본 제안** | ✅ surprise | ✅ surprise·graded | ✅ 공유 | **★ 빈 칸** |

**핵심 관찰**: 축 A·B·C는 각각 이미 채워졌다(= 단독으로는 incremental). 그러나 **D열(하나의 신호로 A+B 동시 구동)은 4편을 직접 정독한 결과 완전히 비어 있다.** "침묵으로부터의 추론"이 아니라 확인된 공백이다.

## 3. 핵심 기여 (Contributions)

1. **통합 (C1)** — 하나의 surprise 신호가 깊이 halting과 fast-weight 갱신을 동시에 구동하는 단일 test-time 루프. *(선점 0%)*
2. **surprise-halting (C2)** — 깊이 정지를 RL(FR-Ponder)이나 KL-수렴(Geiping)이 아닌 surprise로 결정. 
3. **통일 관점 (C3)** — Geiping의 "수렴(변화 멈춤)-halting"과 Titans/PonderTTT의 "surprise-갱신"이 **같은 신호의 양면**임을 드러내고, 그 원리로 둘을 통합. *(이론적 프레이밍)*
4. **상호작용 실증 (C4)** — 고정 test-time FLOPs 예산을 깊이 vs 가중치에 어떻게 배분하는 게 최적인지, 그 최적점이 난이도에 따라 어떻게 이동하는지의 2D frontier. *(누구도 같은 예산선 위에서 두 손잡이를 비교한 적 없음)*

## 4. 방법 (Method)

surprise = "지금 latent가 얼마나 예상 밖인가"(예측/복원 오차). 매 latent 스텝에서 이 신호 하나가 두 결정을 낸다.

```python
z = embed(question)                 # 초기 latent (Coconut식)
W = W_base.clone()                  # 이 쿼리 동안만 사는 fast-weights (episodic)
for t in range(MAX_STEPS):
    z = latent_step(z, W)           # 연속 사고 1스텝 (단어 디코딩 없음)
    s = surprise(z, W)              # self-supervised 오차 = 공유 신호

    W = W - lr * g(s) * grad(s, W)  # [B] surprise에 비례해 갱신 (graded, TTT)
    p_halt = halt_head(z, s)        # [A] surprise 낮으면(수렴) 멈춤
    if sample_halt(p_halt): break
answer = decode(z, W)               # 마지막에만 단어로
```

- `W`는 **episodic fast-weights** — 다음 쿼리에 리셋(continual TTA의 발산 회피).
- 같은 `s`가 **갱신량(B)** 과 **halting(A)** 양쪽에 들어가는 것이 이 방법의 핵심 커플링.
- PonderTTT 대비: 이진 on/off·1스텝이 아니라 **graded 갱신**; 게다가 **깊이 축과 결합**.

## 5. 가설 (Hypotheses)

- **H1 (대체)**: 가중치 적응을 켜면 같은 정확도를 **더 적은 latent 스텝**으로 도달한다.
- **H2 (커플링)**: 고정 FLOPs 예산 하에서 깊이 vs 가중치의 **최적 배분점이 난이도에 따라 이동**한다. → 대표 그림.
- **H3 (신호 유효성)**: surprise가 실제 정답 확률/난이도와 **상관**되고, surprise-halting이 RL·KL-수렴 halting과 **동등 이상** 효율을 낸다.

## 6. 실험 설계 (Experiments)

- **모델**: 1B급 (Qwen2.5-1.5B / Llama-3.2-1B) — 서버 GPU 1~4장 스케일.
- **데이터**: ① 합성 산술(난이도=스텝수 직접 통제 → 커플링 깨끗이 관찰) → ② GSM8K → ③ MATH500.
- **Ablation**:
  1. 고정깊이 Coconut *(baseline)*
  2. + halting만 *(깊이 축; FR-Ponder류)*
  3. + TTT만 *(가중치 축; PonderTTT류)*
  4. **둘 다 · surprise 공유** *(본 제안)*
- **지표**: 정확도 **vs test-time FLOPs**(대표 곡선), surprise–정확도 상관, 난이도별 깊이·갱신 배분(H2 2D frontier).
- **킬러 결과**: (4)가 동일 FLOPs에서 정확도 최고 + H2 배분 지도.
- **전이(P1 차별화)**: 위 4논문 전부 텍스트 추론만 다룸 → **과학·시계열 도메인(PEMFC 등)으로 이식**해 일반성 입증. *(랩 데이터 해자)*

## 7. 예상 반론과 방어 (Risks & Rebuttals)

| 반론 | 방어 |
|---|---|
| "적응형 latent 깊이는 이미 done" | 우리 기여는 깊이가 아니라 **통합(C1)+상호작용(C4)**. 깊이는 구성요소일 뿐. |
| "왜 surprise가 RL/KL보다 나은 halting?" | H3에서 직접 비교. surprise는 **갱신과 신호를 공유**해 추가 비용 없이 두 손잡이를 얻음. |
| "통합 이득이 진짜냐, 그냥 두 개 켠 거냐?" | H2의 2D frontier가 **곱셈적 이득**(단순 합 초과)을 보이면 정당화. |
| "루프 안 가중치 갱신은 발산" | episodic 리셋 + 작은 lr + 갱신 clip. 발산율을 실험에서 측정·보고. |
| 선점 위험: PonderTTT가 future work로 GSM8K 명시 | **속도가 관건.** 통합+전이로 델타를 넓혀 단순 확장과 차별화. |

## 8. 로드맵 (Roadmap)

- **1단계 (안전, ~3주)**: 합성 산술에서 4-way ablation. surprise-halting(C2) + graded TTT(B) 각각 동작 확인. → 워크샵 감.
- **2단계 (도전, ~4주)**: GSM8K/MATH500으로 확장, H2 2D frontier 완성. → 본 논문 코어.
- **3단계 (해자, 유동적)**: 과학·시계열 도메인 전이. → 차별화 섹션.

## 9. 참고문헌 (검증된 arXiv · 3-0)

- Coconut — Training LLMs to Reason in a Continuous Latent Space. **arXiv:2412.06769** (Meta, 2024)
- FR-Ponder — Learning to Ponder: Adaptive Reasoning in Latent Space. **arXiv:2509.24238** (2025)
- Recurrent Depth — Scaling Test-Time Compute with Latent Reasoning. **arXiv:2502.05171** (Geiping 외, 2025)
- PonderTTT — When to Ponder: Adaptive Compute via Test-Time Training. **arXiv:2601.00894** (Gihyeon Sim, 2026-01)
- Titans — Learning to Memorize at Test Time. **arXiv:2501.00663** (Google, 2024)
- TTT layers — RNNs with Expressive Hidden States. **arXiv:2407.04620** (Sun 외, 2024)
- Test-Time Training (원조). **arXiv:1909.13231** (Sun 외, ICML 2020)
- PonderNet — Learning to Ponder. **arXiv:2107.05407** (Banino 외, DeepMind, 2021)

---
*미확인 세부(원문 직접 재확인 필요): Coconut 커리큘럼 학습 스케줄, Titans surprise 수식·메모리 갱신 규칙. 3-pass는 [[research-methodology]] 적용.*

---

## 부록 — 2026-07-06 문헌 재검증 및 주장 정정

이 제안서는 2026-07-03 시점의 스냅샷으로 보존한다. 이후 재검증에서 다음 정정이
필요함이 확인됐다 (README/PROJECT/RESULTS에 반영됨):

1. **§2 "D열은 완전히 비어 있다" → 좁혀야 함.**
   - **UT-Memory (arXiv:2604.21999, 2026-04)** — ACT halting + 학습형 메모리를
     한 블록에 넣고 깊이–메모리 대체(halt 11.6→8.3, 정확도 고정, Sudoku-Extreme)를
     보임. 단, 메모리가 *학습 시점의 아키텍처 용량*(메모리 크기별 별도 학습)이고
     halting이 *학습된 ACT 라우터*라서 본 제안의 test-time 축과는 다름 — 그러나
     **C4("같은 예산선에서 두 손잡이 비교는 전무")는 문장 그대로는 이제 성립하지
     않으므로 철회**하고, 차별화(test-time vs train-time, 신호 vs 라우터)로 대체.
   - **HRM (2506.21734) · TRM (2510.04871)** — 소형 재귀 모델 + 학습된 Q-halting +
     지속 상태로 합성 추론(Sudoku/Maze/ARC)에서 강한 결과. 같은 실험 틈새를 선점
     하고 있으므로 관련 연구에 필수 인용.
   - **살아남은 좁은 빈 칸(bake-off로 더 좁혀짐)**: *TTT/메모리 모듈의 자체
     dynamics가 halting 신호가 되는 구성*(별도 라우터·Q-head·RL 불필요). 2026-07-06
     검색 기준 선행 없음. bake-off(10 seeds): raw recon 손실 자체는 halting에서
     **패배**(−5.6pp), 그러나 latent-step 수렴 노름(dstate≈‖∇_s L‖)은 **정확도 무손실**
     (단 보수적 — 거의 halt 안 함; compute 절감은 conv만) → 빈 칸은 "raw miss"가 아니라
     *내부 최적화의 수렴*이 깊이 신호가 되는 구성으로 좁혀짐.
2. **C3 "같은 신호의 양면" — 전제가 아니라 가설.** 이 동일성은 latent step이
   메모리 손실에 대한 (근사) 경사 하강일 때만 성립한다. 반례 두 방향: 수렴했지만
   surprise 높음(비가역 오차·noisy-TV — 영원히 halt 안 함), surprise 낮지만 계산
   미완(예측 가능한 multi-hop 체인 — 조기 halt). 후자는 Part 3에서 *entropy*
   halting으로 관측(초기 −9.6pp, curriculum+aux·10시드 재측정 −5.6pp) — **단
   bake-off가 이는 entropy/recon 신호 한정임을 확정**(conv/dstate는 −0.0pp) →
   조기-halt 반례는 mechanism이 아니라 신호 선택 문제로 귀결. 개념 계보로 predictive
   coding / free energy(Rao & Ballard 1999; Friston)를 인용할 것 — "하나의 예측
   오차가 추론 정착과 시냅스 갱신을 모두 구동"은 이 전통의 아이디어이며, 소유하면
   강점이고 숨기면 약점.
3. **§6 대비 실제 실행 규모**: 1B 모델·GSM8K·H2 frontier는 미착수. 현재 증거는
   0.2–0.9M 파라미터 합성 과제의 메커니즘 파일럿. 위생: **bake-off(Part 4)·reachp2는
   10 seeds + held-out tau로 해소**, Parts 1–2는 여전히 단일 시드·eval-batch tau(잔여).
   다음은 외부 legible 과제(MQAR)로의 전이 검정 (PROJECT.md §7).
