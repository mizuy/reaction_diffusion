# 二場活性化・抑制モデルの初回結果

2026-09-15。[concept §12](pit_mechanochemical_transition_concept.md)で提案した二場モデルを実装し、独立ランダム初期化から初回の小規模計算を行った。

## 実装

`src/simulate_pit_activator_inhibitor.py` は

\[
\partial_t u=D_u\Delta u+ru-u^3-gv+b,
\qquad
\tau\partial_t v=D_v\Delta v+u-v
\]

を周期領域で半陰的に積分する。各b・各seedは一様定常解に独立な白色ノイズを加えて開始し、前の場を引き継がない。既定値はDu=1、Dv=20、r=1、g=1.4、tau=0.5、初期ノイズ標準偏差0.05。提案時のtau=5では一様モードが不安定だったため、線形安定条件を満たす0.5へ修正した。

一様場のコントラストが0.1未満ならuniformとし、丸め誤差を閾値で二分して模様と数えない。線形Turing条件、場、指標、画像、seed別集計を保存する。

## 初回計算

64×64、dx=1、dt=0.08、5000 step、seed 1/7/42、b=-0.22,-0.18,-0.14,-0.10,-0.06,-0.02,0を計算した。これは探索用smokeで、20 seed・長時間・格子収束の本検証ではない。

- b=-0.22では3 seedすべてI。丸い孤立spotが独立初期化から再現した。
- b=-0.18では3 seedすべてIで、IからIIILへの境界はこの粗い刻みでは確定しない。
- b=-0.14ではseed 1/7がIIIL、seed 42は未分類。短管化する傾向はあるが、共通区間の基準を満たさない。
- b=-0.10〜-0.06では長い非分岐成分が増え、多くが未分類。
- b=-0.02〜0ではseedにより未分類またはIV_V。連結迷路への移行はあるが境界が揺れる。
- IV_Bと分類できる有限分岐は一例もなかった。

幅中央値は模様域で概ね6.4〜8.8格子で、bに伴う変化は比較的小さい。一方、最後20%の場のRMS変化は多くの点で約0.02〜0.09あり、5000 step時点で定常とは判定しない。

## 暫定判断

この最小二場系は、履歴なしでspotから短管・長管・迷路へ移る基本系列を示した。しかし目標の核である有限分岐IV_Bは得られておらず、現時点では四形態モデルとして不合格である。bの細分化と長時間化は必要だが、狭い過渡帯を探して成功扱いにはしない。有限分岐を安定化する機構がこの二場系の係数範囲に存在するかを先に探索する。

出力: `artifacts/pit_independent_initialization/activator_inhibitor_smoke/`。再現例:

```bash
.venv/bin/python src/simulate_pit_activator_inhibitor.py \
  --output artifacts/pit_independent_initialization/activator_inhibitor_smoke \
  --size 64 --steps 5000 \
  --biases -.22 -.18 -.14 -.10 -.06 -.02 0 \
  --seeds 1 7 42
```
