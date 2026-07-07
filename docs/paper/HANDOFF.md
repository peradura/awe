# 논문 작성 핸드오프 (새 세션용)

**시작 프롬프트 예시**: "AWE 워크샵 논문 작성 이어가자 — docs/paper/draft.md 스켈레톤과
HANDOFF.md 읽고 진행."

## 상태 (2026-07-07)
- **v1 산문 완성**: `docs/paper/draft.md` — Abstract→Conclusion 전 섹션을 스켈레톤
  불릿에서 발행 가능한 산문으로 확장. Table 1(4c)·Table 2(Part 5 multi-hop) 본문 삽입.
  모든 헤드라인 수치를 소스 JSON에서 재검증(2026-07-07): Part 4c(conv 71.2±0.5 @
  3.89±0.09, tau_ok 10/10, prem 0.02, null 63.8→+7.4pp; 나머지 4신호 0/10),
  Part 5 multi-hop(conv 42.3 vs ent 39.5, +2.76±0.35pp 10/10, depth 2.75→3.51),
  단일홉 cost-free — 전부 JSON과 일치.
- 스켈레톤(제목후보·그림/표 계획·TODO·writing conventions)은 파일 하단에 보존.
- 방향 근거: codex+critic 교차리뷰 수렴 — "추가 실험 대신 논문 먼저, backfill은
  리뷰어 gap 2개만". 판정 기록: `docs/REVIEW.md`, `docs/exp_logs/LOG.md` 메모,
  `PROJECT.md` §7.

## 그라운드 트루스 (인용할 수치의 원천 — 문서 아닌 JSON 기준)
- 정준 bake-off: `results/bakeoff_reachp2_s{0..9}.json` (Part 4c)
- 외부 전이: `results/mqar_seed*.json`, `results/mqarhop_seed*.json` (Part 5)
- 역사/견고성: `results/reachp3_seed*.json` (4a), sweep 로그 (4b)
- 서술 원천: `docs/RESULTS.md` (Part 4a/4b/4c/5), `PROJECT.md` §4

## 절대 규칙 (이미 리뷰에서 확정된 것 — 어기면 안 됨)
1. 두 recon 정의(state-mismatch vs decodability)를 절대 합치지 말 것.
2. 4a/4b/4c 표를 한 표로 합치지 말 것 — 인용은 4c(정준)로.
3. "one signal, two knobs"를 살아있는 주장처럼 쓰지 말 것 — halting 축에서
   검정·기각된 사실로 서술 (이게 논문의 정직성 자산).
4. write-magnitude probe는 본문 주장 금지 (2중 confound — 후속 논문 시드).
5. 모든 수치 mean±std·10 seeds·held-out tau 표기, 예외는 명시.

## 남은 작업 순서 (갱신 2026-07-07 오후)
1. ~~draft.md 섹션 살 붙이기~~ ✅ v1 산문 완성 + **critic·codex 이중 교차검토 반영**
   (§4.2를 4c 자체 분해수치로 재앵커, recon 태깅 전수화, "weak unification survives"
   문구 제거, conv/dstate 컴퓨트 구분, ±std 보강, std=population 컨벤션 명문화).
   codex 아티팩트: `.omc/artifacts/ask/codex-you-are-cross-reviewing-*.md`
2. ~~Fig 1 multi-seed band~~ ✅ `scripts/fig1_bakeoff_band.py` →
   `results/fig1_bakeoff_band.{png,pdf}` (10-seed 밴드, CVD-safe, grayscale-safe).
3. backfill 2개 — **실행 중(2026-07-07)**: depth_sanity 10시드(GPU0,
   `results/depth_sanity_s*.log`), 표준 MQAR 앵커 n=8192/m=4/Q=16 10시드(GPU1,
   `results/mqar8k_*`; 설계는 `docs/mqar_design.md` §Standard-config anchor).
   완료 후: depth_sanity corr 집계 → draft Limitations 갱신; mqar8k 집계 → §4.3/
   Limitations 갱신 (결과별 해석 시나리오는 설계 섹션에 기록됨).
4. ~~LaTeX 포팅~~ ✅ `docs/paper/latex/` — main.tex(NeurIPS 2024 preprint 스타일,
   DeltaNet arXiv 소스에서 정품 .sty 추출) + references.bib + Fig1/Tab1/Tab2,
   `latexmk -pdf main.tex`로 8p 컴파일 확인(경고 0). 제목 확정: "Convergence,
   Not Surprise: ..." (사용자 선택). ~~① mqar8k Limitations 교체~~ ✅ (전이 성립 —
   conv 2.24±0.16/6스텝 curve read, 10/10; argmax-tau 아티팩트 8192 재현 기록), ~~② 부록 A/B/C~~ ✅ 포팅 완료(signal inventory + 4a/4b 표 +
   tau-규칙 ablation 표, 9p 컴파일 확인), ~~③ bib TODO-VERIFY~~ ✅ arXiv API로
   3건 검증 완료(별칭≠실제 제목 — bib·본문 정정), ④ venue 확정
   후 스타일 파일 교체 + 페이지 fit, ⑤ 저자/소속/지도교수 확정.
5. backfill 2종 모두 ✅ — depth_sanity(corr +0.997±0.001, 구 +0.92 supersede),
   mqar8k 앵커(전이 성립, 판별력은 단일홉이라 없음 — mqar_design.md §anchor result).
   잔여 optional: multi-hop을 vocab 8192로 재실행(판별 claim의 표준 스케일 검증).
