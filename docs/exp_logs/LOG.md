# 실험 로그 인덱스

자동/수동 기록: 실행 로그는 `results/`, 아래 표는 인용할 만한 실험의 한 줄 인덱스.
형식·규칙은 [`LOGGING.md`](../../LOGGING.md) 참고. (판정: ✅ positive / 🟡 directional / 🔴 negative)

| 날짜 | 실험 (module) | 컨셉 | 결과 | 산출물 |
|------------|---------------|------|------|--------|
| 2026-07-03 | `depth_sanity` | in-context reachability로 **깊이 손잡이** sanity | ✅ 깊이↑→정확도↑(r1=64%→r6=100%), `corr(K, 수렴스텝)=+0.92` — halting=사후 예측-안정성(수렴), surprise 아님 | — (로그 미보존 — 재생성 필요) |
| 2026-07-03 | `ablation_ttt` | **공유 신호**(recon error가 양쪽 구동) fast-weight 모델 + 4-way ablation (reachability) | 🟡 배관 검증(과제가 너무 쉬워 config 구분 약함) — **논지 그대로의 구성은 이 실험뿐, 미결** | (미보존 — `accuracy_vs_steps.png` 커밋 안 됨) |
| 2026-07-06 | `ablation_amort` | full-table reachability에 query 스트림+persist 메모리 | 🔴 **negative** — 전부 100%·amortization 평평(그래프가 context에 있어 메모리 잉여). 주의: 단일 tau 공유 → `+halt` 평평함은 일부 tau-스케일 아티팩트 | (로그 미보존) |
| 2026-07-06 | `ablation_rule` | **hidden permutation + 부분관측** 스트림 (메모리 손잡이) | ✅ **memory-knob positive** — capability gap persist 5%→81%(both 5→79%), amortization 8→1.2스텝, `corr(ans, 엔트로피)=−0.96`, both=persist 정확도를 2.4 vs 8스텝. **halting=엔트로피, 쓰기=recon error(두 신호)** | `results/rule_curve.png`, `results/hidden_rule_run.log` |
| 2026-07-06 | `ablation_reachp` | **부분관측 reachability** — 깊이+메모리 결합 stress-test | 🔴 **joint negative / 🟡 memory-only** — persist 35.3%(22%→41%)인데 **both 25.7%(−9.6pp): halting을 켜면 손해**; +halt 16.0% < fixed 21.0%; K≥2 예산 포화. `corr=−0.23`은 라벨링 아티팩트로 과소평가(아래 행) | `results/reachp_curve.png`, `results/reachp_run.log` |
| 2026-07-06 | `reachp.py` 라벨링 수정 + 재실행 | `ans`가 현재 쿼리 공개분을 무시(모델은 retrieval 전에 write) → 라벨 교정 후 동일 시드 재실행 | ✅ 학습·정확도 완전 재현(loss·acc 동일), **corr −0.229 → −0.293** — 아티팩트는 약한 결합의 일부만 설명; joint 과제의 약한 신호-답변가능성 결합은 실재 → bake-off 필요성 강화 | `results/reachp_run_v2.log`, `results/reachp_curve_v2.png` |
| 2026-07-06 | 문서 재캘리브레이션 (외부 리뷰) | 신호 인벤토리 명시(엔트로피 vs recon error), Part 3 both-vs-persist 손실 명시, UT-Memory/HRM/TRM 인용, C4 철회, 한계 섹션 추가 | — | `docs/RESULTS.md` "Signal inventory", `docs/REVIEW.md`, `docs/proposal.md` 부록 |
| 2026-07-06 | `ablation_reachp2` | 개선판: K-curriculum + aux next-node loss + d=256 (GPU0, 8k) | 🟡 **부분 성공** — curriculum+aux가 base-learner 병목 해소: persist 35→**72%**, amortization 40→**79%**. 그러나 **halting을 켜면 여전히 정확도 손실**(both 44.6% < persist 72.2%), depth∝K 조잡(K≥1 증가하나 예산 포화, K=0 이상치). ⚠️ `corr=−0.33`은 **pre-ans-fix라 재실행 필요**. 진단 확정: **base-learner를 고쳐도 halting-신호가 병목** → bake-off로 직행 | `results/reachp2_curve.png`, `results/reachp2_run.log` |
| 2026-07-06 | held-out tau + `bakeoff` + 러너 | ① `calib_tau`를 held-out 배치로 (rule/reachp/reachp2), ② halting-신호 bake-off 스크립트(ent/recon/rnorm/dstate/dent + shuf_g/shuf_b 대조 + tau-스윕 곡선 + 고정깊이 frontier + 실패분해 + corr), ③ 다중시드 스윕(ckpt 재사용)·집계·공유GPU 예절 러너 | ⏳ 실행 중 — 서버2: reachp2(교정판)+seed 0–4, 컨테이너: seed 5–9 | `experiments/bakeoff.py`, `scripts/{sweep.sh,aggregate.py,gpu_watch_run.sh}` |

## 메모
- **핵심 전환점**: `ablation_amort`(negative) → 진단(그래프가 context에 있어 메모리 잉여, shortcut 없음) → `ablation_rule`(부분관측으로 메모리 필수화) = memory-knob positive. 단, 메모리가 필수인 과제에서 persist가 reset을 이기는 것은 상당 부분 과제 설계의 귀결 — 검증된 것은 하네스+배관.
- **깊이(1) · 메모리(2)는 각자 다른 신호로 깨끗, 결합(3)은 negative** — 원인은 세 갈래로 교란: (a) base-learner의 memory-chain-following 병목, (b) `ans` 라벨링 아티팩트(수정됨), (c) 엔트로피-임계값 halting의 multi-hop 실패(확신-오답 조기 종료 = both가 persist보다 −9.6pp). 이전의 "mechanism 문제 아님" 단정은 근거 부족이라 철회.
- **reachp2가 (a)와 (c)를 분리 확정**: curriculum+aux로 base-learner를 크게 개선(persist 35→72%)해도 **halting을 켜면 여전히 손해** → 남은 병목은 (c) **halting-신호 자체**임이 확정됨.
- **미실행 핵심**: 논지("recon error 하나가 양쪽 구동")를 capability gap 있는 과제에서 돌린 적이 아직 없음. → 다음은 halting-신호 bake-off(recon vs 엔트로피 vs Δstate vs 랜덤-매칭 vs 고정깊이-매칭, held-out tau, ≥5 seeds) + halt 실패 분해(조기-오답/조기-정답/미확신).
- 이후: MQAR/in-context regression으로 대외 검증. 중단 기준은 PROJECT.md §7.
