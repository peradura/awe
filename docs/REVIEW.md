# 외부 리뷰 기록 — 2026-07-06

멀티-에이전트 교차 검증 리뷰(방법론 / 코드 감사 / 문헌·novelty 공격 + 상호 반박
라운드)의 요약. 문서 정정의 근거이자, 다음 실험 설계의 입력. 원 수치는 모두
`results/*.log`와 대조 검증됨.

## 판정 요약

**방향은 유지할 가치가 있고 부정-결과 위생은 평균 이상. 그러나 간판 주장
"one surprise signal, two knobs"는 세 실험 중 어느 것으로도 아직 뒷받침되지
않는다** — 긍정 실험들은 두 개의 서로 다른 신호(쓰기=recon error,
halting=엔트로피)를 쓰고, 논지 그대로의 구성(`ttt.py`)이 돌아간 과제는
미결/negative였다.

## 확정된 발견 (심각도순)

1. **주장-구현 불일치 (critical)** — README/PROJECT의 "재구성 오차 하나가 양쪽
   구동"은 `models/memory.py`(긍정 결과의 모델)와 불일치: halting은 readout
   엔트로피(`memory.py` retrieve), 쓰기는 delta-rule recon error(write).
   → 문서 재캘리브레이션 완료(2026-07-06); 공유-신호 실험은 로드맵 최상단.
2. **Part 3 선택적 보고 (critical)** — `reachp_run.log`: persist 35.3% vs
   both 25.7%(−9.6pp), +halt 16.0% < fixed 21.0%. 기존 문서는 persist의
   22→41%만 "joint" 행에 인용. → RESULTS/PROJECT/LOG에 명시 완료. 지배적 실패
   모드는 확신-오답 조기 종료(both가 5.4스텝에서 멈추며 손해).
3. **`reachp.py` 라벨링 아티팩트 (major)** — 모델은 현재 쿼리 엣지를 retrieval
   전에 write하는데 `ans`는 strictly-prior만 인정 → corr(−0.229) 과소평가,
   fixed 베이스라인(21%) 부풀림. → 코드 수정 + 동일 시드 재실행 완료
   (`reachp_run_v2.log`): 학습·정확도 완전 재현, **corr −0.229 → −0.293** —
   아티팩트는 일부만 설명, joint 과제의 약한 결합은 실재로 확정.
4. **방법론 위생 (major)** — 전 실험 단일 시드·오차막대 없음; `calib_tau`가
   평가 배치에서 캘리브레이션(테스트-셋 누수); "3.3×"는 retrieval 스텝만 집계
   (write 비용 제외); `ablation_amort`는 단일 tau 공유(+halt에 불공정);
   `depth_sanity` 로그 미보존. → 한계 섹션으로 문서화, bake-off 재실행에 귀속.
5. **문헌 포지셔닝 (major)** — UT-Memory(2604.21999, 2026-04)가 깊이–메모리
   대체를 선행(단 train-time 용량 + ACT 라우터 — test-time 논지는 선점 아님);
   HRM(2506.21734)/TRM(2510.04871)이 같은 실험 틈새 선점. C4는 철회.
   **살아남은 좁은 빈 칸: TTT 손실 자체가 halting 신호(라우터 불필요)** —
   2026-07-06 검색 기준 선행 없음. 개념 계보로 predictive coding 인용 필요.

## 토론 라운드에서 결정된 것

- corr −0.96→−0.23 하락을 "이론적 해리의 실증"으로 읽는 해석은 **철회**
  (라벨링 아티팩트 + K≥2 포화는 "미완료를 정확히 보고하는 신호"로도 설명됨).
  해리의 증거는 아티팩트가 닿지 않는 both-vs-persist −9.6pp로 **재배치**.
- Δsurprise(≈0이면 halt) 제안은 multi-hop 실패를 못 고침(확신-오답이면 낮고
  평평 → 역시 조기 halt): **단독 해법이 아니라 bake-off의 한 팔**로 강등.
  noisy-TV 케이스에는 유효.
- "같은 신호의 양면"이 정리(theorem)가 되는 유일한 경로: latent step을 메모리
  손실에 대한 경사 하강으로 정의(‖Δs‖ ∝ ‖∇L‖) — 구조적 통합의 후보.
- **다음 최고 정보량 실험**: halt 실패 분해 표(조기-오답/조기-정답/미확신) —
  "tau 미캘리브레이션 / base-learner 한계 / halting 의미론 오류"를 갈라줌.

## 실행 순서 (합의안)

1. `reachp.py` 라벨링 수정 + 재실행 (✅ 이 커밋)
2. 문서 재캘리브레이션 (✅ 이 커밋)
3. halting-신호 bake-off + 실패 분해 (held-out tau, ≥5 seeds) — PROJECT.md §7
4. 공유 recon-error 신호를 Part 2–3에 실제 적용 (남은 novelty의 본체)
5. 중단 기준: PROJECT.md §7의 kill criterion
