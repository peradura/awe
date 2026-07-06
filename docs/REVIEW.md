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
   전에 write하는데 `ans`는 strictly-prior만 인정 → corr(−0.229) 과소평가.
   (fixed 21%가 우연 수준 1/24를 상회하는 것은 라벨과 무관한 별도 현상 —
   현재-쿼리로 풀리는 probe + sink probe + sink prior로 설명되며, 정확도
   수치 자체는 라벨 수정의 영향을 받지 않음.) → 코드 수정 + 동일 시드 재실행 완료
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

1. `reachp.py` 라벨링 수정 + 재실행 (✅)
2. 문서 재캘리브레이션 (✅)
3. halting-신호 bake-off + 실패 분해 (held-out tau, ≥5 seeds) — ✅ **완료(10 seeds)**, 아래 판정
4. ~~공유 recon-error 신호를 Part 2–3에 실제 적용~~ — bake-off가 **recon halting 반증**으로 대체(강한 통합 미지지)
5. 중단 기준: PROJECT.md §7의 kill criterion (수렴-halting의 MQAR 전이 검정)

---

## 2026-07-06 후속: bake-off 판정 + 재검증

**bake-off 판정 (`ablation_reachp3`, 10 seeds, held-out tau)** — persist 천장 72.0±0.6%:
- **conv·dstate 무비용**(gap −0.0pp): conv 5.00스텝·early-right 34%(compute 효율 최선); dstate(=‖Δs‖²) early-wrong 1.1%(가장 안전)하나 보수적(5.93스텝).
- **entropy·recon −5.6pp**(early-wrong 8–9%, corr(ans) −0.45/−0.25) → 기존 halting 손실 = **신호 선택 오류**, mechanism 실패 아님.
- 논지 정정: 강형("한 스칼라가 양쪽 *최적* 구동") **죽음** — 효율 최선 conv는 write 역할 없음; recon(논지 신호)은 halting 패배(테스트한 정의+held-out tau 기준); dstate는 정확도는 보존하나 보수적(거의 halt 안 함). → **깊이·write는 서로 다른 observable을 원함**. "두 gradient=통합"은 동어반복적 *해석*일 뿐, dstate↔write크기 예측을 실증해야 비자명(MQAR probe).
- reachp2 다시드(10 seeds): persist 72.1±0.6%(reset 43.3%→persist), corr(ans,sur0) −0.452±0.010 — base-learner fix 견고. `both`(entropy·median-tau) 44.9%는 tau 미최적화 아티팩트.

**architect 재검증 (다음 실험 방향)** — 원래 후보였던 **GD-step "by construction" 통합실험은 비추천**으로 결론:
- (a) `‖Δs‖=η‖∇L‖`는 알고리즘 항등식 → 대부분 **동어반복**(남는 실증 내용은 좁은 비열등성/용량유지뿐).
- (b) write는 쿼리당·`∇_W L`, halt는 스텝당·`∇_s L`로 granularity가 달라 **"하나의 스칼라" write-tie가 비정합** — 억지로 묶으면 `ttt.py`(episodic)가 되어 Part 2 amortization 상실.
- (c) inference-valid한 `target(s)`는 fixed-point residual뿐 → halt 신호가 이미 진 **recon으로 붕괴** 위험.
- (d) 데이터 확인: 10시드에서 최고 신호는 dstate가 아니라 **conv**(효율)인데 통합 서사 없음 → 강한 프레이밍 자체가 약함.
- **채택안**: 기존 conv/dstate 결과를 **정직하게 재프레이밍**(강한 통합 죽음; 깊이·write는 서로 다른 observable; "통합"은 미검정 해석; 완료) + GPU는 **외부 legible 과제(MQAR 1순위)**로 수렴-halting 전이 검정. GD-step은 fallback(리뷰어가 exact identity 요구 시).

*교차검증*: 위 재프레이밍 claim은 `/ask codex` 교차검토 대상(사용자 규약) — 별도 검증 패스에서 수행.
