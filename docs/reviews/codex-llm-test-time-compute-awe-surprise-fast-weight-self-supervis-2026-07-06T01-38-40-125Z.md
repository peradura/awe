# codex advisor artifact

- Provider: codex
- Exit code: 0
- Created at: 2026-07-06T01:38:40.125Z

## Original task

LLM test-time compute 연구 파일럿의 실험 설계에 대한 비판적 리뷰가 필요해.

## 아이디어 (AWE)
하나의 "surprise" 신호(fast-weight 연상메모리의 self-supervised 재구성 오차)가 test-time에 (A) latent 추론 깊이 halting(surprise 낮으면 멈춤)과 (B) fast-weight TTT 갱신(surprise 비례)을 동시 구동. 문헌상 이 통합은 미개척(FR-Ponder=RL 깊이halting, Geiping=KL수렴 halting, PonderTTT=이진 weight게이팅, Titans=surprise 메모리 — 아무도 한 신호로 A+B 동시 안 함).

## testbed
함수그래프 도달성: 입력에 f:[n]->[n] 테이블 전체 + 시작노드 v0. f 따라가 sink(고정점) 찾기. 난이도 K=sink까지 홉수. 매 step t readout을 f^t(v0)로 지도학습. 소형 recurrent 모델(~0.76M).

## amortization 실험 결과 (negative)
한 그래프에 여러 query 스트림, 메모리 persist. 기대: 뒤 query일수록 적은 홉으로 halt. 결과: 전부 정확도 100%. steps/query: fixed/+halt/+amort=12(고정), both=6.95지만 스트림 내내 평평(q0=7.2->q7=6.8, 노이즈). amortization 우하향 안 나옴.

## 내 진단
1) 그래프 전체가 매 query context에 이미 있어 메모리 잉여. both 메모리도 첫 query 안에서 이미 적응(q0=7)해 누적될 게 없음.
2) tau를 both median으로 보정->+halt엔 불리(confound). both=7도 구성상 인공물 의심.
3) 과제 너무 쉬움(전부 100%)->capability headroom 없음.
4) 답 f^K는 정확히 K홉 필요+랜덤그래프라 shortcut 없음->weight가 깊이 대체(H1) 구조적 불가.

## 3갈림길
1) 그래프를 매 query에 부분만 노출(메모리가 조각 누적) 2) 명시적 sink-cache+capability gap 3) testbed 근본 재고

## 질문
(a) 진단 4개 타당? 놓친 confound/대안해석?
(b) 3갈림길 중 어디? 더 나은 4번째 설계? "surprise가 depth+weight 동시 구동"의 가치를 깨끗이 보여줄 minimal task는? (parity/associative recall/copy/in-context regression 등 TTT가 이미 이득 보이는 과제 고려)
(c) reachability 버리고 TTT literature 검증 과제(long-context associative recall, MQAR 등)로 가는 게 나은가?
간결하고 구체적으로, 실행가능한 추천으로.

## Final prompt

LLM test-time compute 연구 파일럿의 실험 설계에 대한 비판적 리뷰가 필요해.

## 아이디어 (AWE)
하나의 "surprise" 신호(fast-weight 연상메모리의 self-supervised 재구성 오차)가 test-time에 (A) latent 추론 깊이 halting(surprise 낮으면 멈춤)과 (B) fast-weight TTT 갱신(surprise 비례)을 동시 구동. 문헌상 이 통합은 미개척(FR-Ponder=RL 깊이halting, Geiping=KL수렴 halting, PonderTTT=이진 weight게이팅, Titans=surprise 메모리 — 아무도 한 신호로 A+B 동시 안 함).

## testbed
함수그래프 도달성: 입력에 f:[n]->[n] 테이블 전체 + 시작노드 v0. f 따라가 sink(고정점) 찾기. 난이도 K=sink까지 홉수. 매 step t readout을 f^t(v0)로 지도학습. 소형 recurrent 모델(~0.76M).

## amortization 실험 결과 (negative)
한 그래프에 여러 query 스트림, 메모리 persist. 기대: 뒤 query일수록 적은 홉으로 halt. 결과: 전부 정확도 100%. steps/query: fixed/+halt/+amort=12(고정), both=6.95지만 스트림 내내 평평(q0=7.2->q7=6.8, 노이즈). amortization 우하향 안 나옴.

## 내 진단
1) 그래프 전체가 매 query context에 이미 있어 메모리 잉여. both 메모리도 첫 query 안에서 이미 적응(q0=7)해 누적될 게 없음.
2) tau를 both median으로 보정->+halt엔 불리(confound). both=7도 구성상 인공물 의심.
3) 과제 너무 쉬움(전부 100%)->capability headroom 없음.
4) 답 f^K는 정확히 K홉 필요+랜덤그래프라 shortcut 없음->weight가 깊이 대체(H1) 구조적 불가.

## 3갈림길
1) 그래프를 매 query에 부분만 노출(메모리가 조각 누적) 2) 명시적 sink-cache+capability gap 3) testbed 근본 재고

## 질문
(a) 진단 4개 타당? 놓친 confound/대안해석?
(b) 3갈림길 중 어디? 더 나은 4번째 설계? "surprise가 depth+weight 동시 구동"의 가치를 깨끗이 보여줄 minimal task는? (parity/associative recall/copy/in-context regression 등 TTT가 이미 이득 보이는 과제 고려)
(c) reachability 버리고 TTT literature 검증 과제(long-context associative recall, MQAR 등)로 가는 게 나은가?
간결하고 구체적으로, 실행가능한 추천으로.

## Raw output

```text
진단 4개는 대체로 타당합니다. 특히 핵심은 **“메모리가 누적할 정보가 없고, weight update가 depth를 대체할 구조도 없다”**입니다.

**(a) 진단 리뷰**

1. **그래프 전체 context 제공 → 메모리 잉여**
   타당합니다. 매 query마다 전체 `f:[n]->[n]`가 다시 보이면 persistent memory는 “새 정보 저장”이 아니라 같은 정보를 재인코딩할 뿐입니다. amortization을 보려면 query 간에 **관측 정보가 비대칭**이어야 합니다.

2. **tau를 both median으로 보정한 confound**
   맞습니다. `both`에 맞춘 halting threshold는 `+halt`를 불리하게 만들 수 있습니다. 비교하려면 각 조건별로 동일 compute budget 또는 validation Pareto frontier에서 비교해야 합니다.

3. **100% 정확도 → headroom 없음**
   매우 중요합니다. accuracy가 saturated면 남는 신호는 steps뿐인데, steps 감소가 “좋은 추론”인지 “threshold artifact”인지 분리하기 어렵습니다. capability gap이 필요합니다.

4. **정확히 K-hop 필요 → TTT가 depth 대체 불가**
   거의 맞습니다. 랜덤 함수그래프에서 sink는 일반적으로 shortcut 구조가 없습니다. fast-weight가 할 수 있는 것은 “이미 본 edge/path/cache 재사용”이지, 처음 보는 경로의 K-hop 계산을 압축하는 것은 어렵습니다. 따라서 H1을 보이려면 **반복되는 substructure, reusable rule, cached intermediate**가 있어야 합니다.

놓친 confound는 세 가지입니다.

- **surprise 신호의 의미 혼재**: reconstruction error가 “novelty”, “uncertainty”, “difficulty”, “OOD”, “memory miss” 중 무엇인지 분리되지 않습니다.
- **within-query adaptation vs across-query amortization 분리 부족**: q0에서 이미 7 step으로 줄었다면, 그 감소가 TTT 때문인지 halting calibration 때문인지 별도 ablation이 필요합니다.
- **halting target leakage 가능성**: step-wise supervision이 강하면 모델이 “현재 노드 추적”은 잘하지만, surprise가 실제 planning depth를 반영하지 않을 수 있습니다.

**(b) 갈림길 추천**

제 추천은 **1번 + 2번의 조합**입니다. reachability를 바로 버리기보다, 한 번 더 “amortization이 보일 수밖에 없는” 형태로 바꿔서 AWE 가설을 깨끗이 테스트하세요.

가장 좋은 minimal variant:

**Partial-observation reachability with reusable path cache**

- 각 graph는 고정.
- query마다 전체 table을 주지 말고, 시작노드 주변의 일부 edge만 제공.
- 모델은 sink를 맞히거나 다음 노드를 예측.
- query들이 같은 graph의 overlapping basin을 공유하게 설계.
- 앞 query에서 본 `(node -> successor)` 또는 `(node -> sink)` 정보가 뒤 query에 도움 되도록 만듦.
- persistent fast-weight memory는 query 간 유지.
- surprise가 높을 때 memory update, 낮을 때 halt.

핵심 metric:

- `query index`에 따른 steps/query 감소
- 동일 compute에서 accuracy 상승
- memory reset vs persist 차이
- surprise와 실제 memory miss/cache hit의 상관
- path overlap 높음/낮음 조건 비교

명시적 sink-cache를 넣는 것도 좋지만, 너무 hand-crafted하면 “fast-weight TTT의 가치”보다 “cache engineering”처럼 보일 수 있습니다. 그래서 처음에는 **implicit fast-weight cache**로 가고, 분석용 oracle sink-cache를 upper bound로 두는 게 좋습니다.

더 나은 4번째 설계는:

**Reusable latent rule task**

TTT가 depth를 대체하려면 “반복 관측으로 latent structure를 학습하면 이후 추론이 짧아지는” 과제가 더 적합합니다.

예시:

- hidden permutation / affine map / modular rule을 여러 query에서 공유
- 초반 query는 rule inference가 필요해서 depth 필요
- fast-weight가 rule을 적응하면 후반 query는 얕게 해결
- surprise는 rule mismatch 또는 memory miss에서 높아짐

예:  
`y = A x + b mod p` 또는 hidden permutation `π`, query는 `π^k(x)` 예측. Context는 일부 pairs만 제공. 같은 episode 안에서 rule이 공유됨. 후반부에 amortization이 자연스럽게 나와야 합니다.

이게 reachability보다 AWE에 더 잘 맞습니다. 이유는 weight update가 단순 cache가 아니라 **episode-specific structure를 압축**할 수 있기 때문입니다.

**(c) MQAR/associative recall로 갈까?**

네, 병행하는 게 낫습니다. 단, reachability를 완전히 버리기보다 역할을 나누세요.

- **Main claim 검증**: MQAR, long-context associative recall, in-context regression류
- **AWE-specific toy**: partial-observation reachability 또는 reusable latent rule
- **Reachability original**: negative result로 보관

가장 실행가능한 순서는 이렇습니다.

1. 먼저 **MQAR/associative recall**에서 AWE가 기존 TTT baseline 대비 compute-accuracy Pareto를 개선하는지 확인.
2. 동시에 **partial-observation reachability**로 amortization curve가 실제로 나오는지 확인.
3. 둘 중 하나라도 “surprise 하나로 halting+update를 동시에 조절했을 때 이득”이 명확하면 그쪽을 메인으로 삼기.
4. original full-table reachability는 “왜 안 되는지”를 ablation/negative section으로 사용.

최종 추천:  
**reachability full-table은 버리고, partial-observation reachability는 진단용으로 유지하세요. 메인 실험은 MQAR/associative recall 또는 hidden-rule regression으로 옮기는 게 더 깨끗합니다.** AWE의 가치는 “어려운 문제를 더 깊게 푼다”가 아니라, **surprise가 memory miss를 감지하고, memory가 쌓이면 depth를 줄인다**로 정의하는 편이 가장 설득력 있습니다.


OpenAI Codex v0.142.5
--------
workdir: /home/dongwan0629/research/llm-surprise-ttt
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: none
reasoning summaries: none
session id: 019f3513-5cba-7ad0-8e8c-7f0fd1fa766f
--------
user
LLM test-time compute 연구 파일럿의 실험 설계에 대한 비판적 리뷰가 필요해.

## 아이디어 (AWE)
하나의 "surprise" 신호(fast-weight 연상메모리의 self-supervised 재구성 오차)가 test-time에 (A) latent 추론 깊이 halting(surprise 낮으면 멈춤)과 (B) fast-weight TTT 갱신(surprise 비례)을 동시 구동. 문헌상 이 통합은 미개척(FR-Ponder=RL 깊이halting, Geiping=KL수렴 halting, PonderTTT=이진 weight게이팅, Titans=surprise 메모리 — 아무도 한 신호로 A+B 동시 안 함).

## testbed
함수그래프 도달성: 입력에 f:[n]->[n] 테이블 전체 + 시작노드 v0. f 따라가 sink(고정점) 찾기. 난이도 K=sink까지 홉수. 매 step t readout을 f^t(v0)로 지도학습. 소형 recurrent 모델(~0.76M).

## amortization 실험 결과 (negative)
한 그래프에 여러 query 스트림, 메모리 persist. 기대: 뒤 query일수록 적은 홉으로 halt. 결과: 전부 정확도 100%. steps/query: fixed/+halt/+amort=12(고정), both=6.95지만 스트림 내내 평평(q0=7.2->q7=6.8, 노이즈). amortization 우하향 안 나옴.

## 내 진단
1) 그래프 전체가 매 query context에 이미 있어 메모리 잉여. both 메모리도 첫 query 안에서 이미 적응(q0=7)해 누적될 게 없음.
2) tau를 both median으로 보정->+halt엔 불리(confound). both=7도 구성상 인공물 의심.
3) 과제 너무 쉬움(전부 100%)->capability headroom 없음.
4) 답 f^K는 정확히 K홉 필요+랜덤그래프라 shortcut 없음->weight가 깊이 대체(H1) 구조적 불가.

## 3갈림길
1) 그래프를 매 query에 부분만 노출(메모리가 조각 누적) 2) 명시적 sink-cache+capability gap 3) testbed 근본 재고

## 질문
(a) 진단 4개 타당? 놓친 confound/대안해석?
(b) 3갈림길 중 어디? 더 나은 4번째 설계? "surprise가 depth+weight 동시 구동"의 가치를 깨끗이 보여줄 minimal task는? (parity/associative recall/copy/in-context regression 등 TTT가 이미 이득 보이는 과제 고려)
(c) reachability 버리고 TTT literature 검증 과제(long-context associative recall, MQAR 등)로 가는 게 나은가?
간결하고 구체적으로, 실행가능한 추천으로.
codex
진단 4개는 대체로 타당합니다. 특히 핵심은 **“메모리가 누적할 정보가 없고, weight update가 depth를 대체할 구조도 없다”**입니다.

**(a) 진단 리뷰**

1. **그래프 전체 context 제공 → 메모리 잉여**
   타당합니다. 매 query마다 전체 `f:[n]->[n]`가 다시 보이면 persistent memory는 “새 정보 저장”이 아니라 같은 정보를 재인코딩할 뿐입니다. amortization을 보려면 query 간에 **관측 정보가 비대칭**이어야 합니다.

2. **tau를 both median으로 보정한 confound**
   맞습니다. `both`에 맞춘 halting threshold는 `+halt`를 불리하게 만들 수 있습니다. 비교하려면 각 조건별로 동일 compute budget 또는 validation Pareto frontier에서 비교해야 합니다.

3. **100% 정확도 → headroom 없음**
   매우 중요합니다. accuracy가 saturated면 남는 신호는 steps뿐인데, steps 감소가 “좋은 추론”인지 “threshold artifact”인지 분리하기 어렵습니다. capability gap이 필요합니다.

4. **정확히 K-hop 필요 → TTT가 depth 대체 불가**
   거의 맞습니다. 랜덤 함수그래프에서 sink는 일반적으로 shortcut 구조가 없습니다. fast-weight가 할 수 있는 것은 “이미 본 edge/path/cache 재사용”이지, 처음 보는 경로의 K-hop 계산을 압축하는 것은 어렵습니다. 따라서 H1을 보이려면 **반복되는 substructure, reusable rule, cached intermediate**가 있어야 합니다.

놓친 confound는 세 가지입니다.

- **surprise 신호의 의미 혼재**: reconstruction error가 “novelty”, “uncertainty”, “difficulty”, “OOD”, “memory miss” 중 무엇인지 분리되지 않습니다.
- **within-query adaptation vs across-query amortization 분리 부족**: q0에서 이미 7 step으로 줄었다면, 그 감소가 TTT 때문인지 halting calibration 때문인지 별도 ablation이 필요합니다.
- **halting target leakage 가능성**: step-wise supervision이 강하면 모델이 “현재 노드 추적”은 잘하지만, surprise가 실제 planning depth를 반영하지 않을 수 있습니다.

**(b) 갈림길 추천**

제 추천은 **1번 + 2번의 조합**입니다. reachability를 바로 버리기보다, 한 번 더 “amortization이 보일 수밖에 없는” 형태로 바꿔서 AWE 가설을 깨끗이 테스트하세요.

가장 좋은 minimal variant:

**Partial-observation reachability with reusable path cache**

- 각 graph는 고정.
- query마다 전체 table을 주지 말고, 시작노드 주변의 일부 edge만 제공.
- 모델은 sink를 맞히거나 다음 노드를 예측.
- query들이 같은 graph의 overlapping basin을 공유하게 설계.
- 앞 query에서 본 `(node -> successor)` 또는 `(node -> sink)` 정보가 뒤 query에 도움 되도록 만듦.
- persistent fast-weight memory는 query 간 유지.
- surprise가 높을 때 memory update, 낮을 때 halt.

핵심 metric:

- `query index`에 따른 steps/query 감소
- 동일 compute에서 accuracy 상승
- memory reset vs persist 차이
- surprise와 실제 memory miss/cache hit의 상관
- path overlap 높음/낮음 조건 비교

명시적 sink-cache를 넣는 것도 좋지만, 너무 hand-crafted하면 “fast-weight TTT의 가치”보다 “cache engineering”처럼 보일 수 있습니다. 그래서 처음에는 **implicit fast-weight cache**로 가고, 분석용 oracle sink-cache를 upper bound로 두는 게 좋습니다.

더 나은 4번째 설계는:

**Reusable latent rule task**

TTT가 depth를 대체하려면 “반복 관측으로 latent structure를 학습하면 이후 추론이 짧아지는” 과제가 더 적합합니다.

예시:

- hidden permutation / affine map / modular rule을 여러 query에서 공유
- 초반 query는 rule inference가 필요해서 depth 필요
- fast-weight가 rule을 적응하면 후반 query는 얕게 해결
- surprise는 rule mismatch 또는 memory miss에서 높아짐

예:  
`y = A x + b mod p` 또는 hidden permutation `π`, query는 `π^k(x)` 예측. Context는 일부 pairs만 제공. 같은 episode 안에서 rule이 공유됨. 후반부에 amortization이 자연스럽게 나와야 합니다.

이게 reachability보다 AWE에 더 잘 맞습니다. 이유는 weight update가 단순 cache가 아니라 **episode-specific structure를 압축**할 수 있기 때문입니다.

**(c) MQAR/associative recall로 갈까?**

네, 병행하는 게 낫습니다. 단, reachability를 완전히 버리기보다 역할을 나누세요.

- **Main claim 검증**: MQAR, long-context associative recall, in-context regression류
- **AWE-specific toy**: partial-observation reachability 또는 reusable latent rule
- **Reachability original**: negative result로 보관

가장 실행가능한 순서는 이렇습니다.

1. 먼저 **MQAR/associative recall**에서 AWE가 기존 TTT baseline 대비 compute-accuracy Pareto를 개선하는지 확인.
2. 동시에 **partial-observation reachability**로 amortization curve가 실제로 나오는지 확인.
3. 둘 중 하나라도 “surprise 하나로 halting+update를 동시에 조절했을 때 이득”이 명확하면 그쪽을 메인으로 삼기.
4. original full-table reachability는 “왜 안 되는지”를 ablation/negative section으로 사용.

최종 추천:  
**reachability full-table은 버리고, partial-observation reachability는 진단용으로 유지하세요. 메인 실험은 MQAR/associative recall 또는 hidden-rule regression으로 옮기는 게 더 깨끗합니다.** AWE의 가치는 “어려운 문제를 더 깊게 푼다”가 아니라, **surprise가 memory miss를 감지하고, memory가 쌓이면 depth를 줄인다**로 정의하는 편이 가장 설득력 있습니다.
tokens used
4,134

```

## Concise summary

Provider completed successfully. Review the raw output for details.

## Action items

- Review the response and extract decisions you want to apply.
- Capture follow-up implementation tasks if needed.
