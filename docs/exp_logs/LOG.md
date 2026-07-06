# 실험 로그 인덱스

자동/수동 기록: 실행 로그는 `results/`, 아래 표는 인용할 만한 실험의 한 줄 인덱스.
형식·규칙은 [`LOGGING.md`](../../LOGGING.md) 참고. (판정: ✅ positive / 🟡 directional / 🔴 negative)

| 날짜 | 실험 (module) | 컨셉 | 결과 | 산출물 |
|------------|---------------|------|------|--------|
| 2026-07-03 | `depth_sanity` | in-context reachability로 **깊이 손잡이** sanity | ✅ 깊이↑→정확도↑(r1=64%→r6=100%), `corr(K, 수렴스텝)=+0.92` | — |
| 2026-07-03 | `ablation_ttt` | surprise-통합 fast-weight 모델 + 4-way ablation (reachability) | 🟡 배관 검증(과제가 너무 쉬워 config 구분 약함) | `results/accuracy_vs_steps.png` |
| 2026-07-06 | `ablation_amort` | full-table reachability에 query 스트림+persist 메모리 | 🔴 **negative** — 전부 100%·amortization 평평(그래프가 context에 있어 메모리 잉여) | `results/` |
| 2026-07-06 | `ablation_rule` | **hidden permutation + 부분관측** 스트림 (메모리 손잡이) | ✅ **first positive** — capability gap 5%→81%, amortization 8→1.2스텝, `corr(miss)=−0.96`, both=persist 정확도를 2.4 vs 8스텝 | `results/rule_curve.png`, `results/hidden_rule_run.log` |
| 2026-07-06 | `ablation_reachp` | **부분관측 reachability** — 깊이+메모리 결합 stress-test | 🟡 **directional** — persist 22%→41%, depth가 K 따라 늘지만 K≥2 예산 포화, `corr(miss)=−0.23`(base-learner 병목) | `results/reachp_curve.png`, `results/reachp_run.log` |
| 2026-07-06 | `ablation_reachp2` | 개선판: K-curriculum + aux next-node loss + d=256 | ⏳ **running** — GPU1 완전히 빌 때 자동 실행 대기 중 | (예정) `results/reachp2_curve.png` |

## 메모
- **핵심 전환점**: `ablation_amort`(negative) → 진단(그래프가 context에 있어 메모리 잉여, shortcut 없음) → `ablation_rule`(부분관측으로 메모리 필수화) = 첫 positive.
- **깊이(1) · 메모리(2)는 각각 깨끗**, **결합(3)은 directional** — base-learner의 memory-chain-following 학습이 병목(mechanism 문제 아님).
- 다음: `reachp2`(curriculum+aux)로 결합을 sharpen, 이후 MQAR/in-context regression으로 대외 검증.
