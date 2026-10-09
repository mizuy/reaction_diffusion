# Pit pattern の細胞力学モデル — CPM / off-lattice 設計メモ

## 1. 目的

[`pit_pattern_goal.md`](../pit_pattern_goal.md) が定める **I → IIIL → IV_B → IV_V** を、濃度波ではなく、上皮の増殖、接着、体積排除、間質の圧排から再現する。

本稿では次の 2 系統を候補とする。

1. **Cellular Potts Model (CPM)**: 細胞を格子サイトの集合として表し、接着・面積エネルギーを Metropolis 法で緩和する。
2. **off-lattice 2 成分粒子モデル**: 上皮 E と間質 S を連続座標上の粒子として表し、Morse 型相互作用と分裂・消失で発展させる。

いずれも Type ごとの独立設定は作らず、単一の進行度 \(p\in[0,1]\) から内部係数を滑らかに決める。成功条件は機構ではなく、既存の幅・分岐・連結性指標と規範画像で判定する。

## 2. 共通仮説

- 低い \(p\): 上皮と間質の体積が釣り合い、孤立した類円形 pit が安定する（I）。
- 中間の \(p\): 上皮の増殖圧が増え、隣接 pit が接近・変形して有限長の短管になる（IIIL）。
- さらに高い \(p\): 圧縮と再配置で局所分岐が生じるが、単位構造はまだ有限長に保たれる（IV_B）。
- 高い \(p\): 間質が細いチャネル／島へ圧排され、上皮–間質界面が連結した等方的迷路になる（IV_V）。

この系列は検証すべき仮説であり、体積ノブだけで必ず成立するとは仮定しない。特に、IIIL を飛ばす相転移、太い blob、平行 stripe、全面占有は失敗とする。

## 3. Cellular Potts Model

### 3.1 状態とエネルギー

格子サイト \(x\) に細胞 ID \(\sigma(x)\) と細胞種 \(\tau(\sigma)\in\{E,S\}\) を持たせる。PoC では E を個別 ID、S を連続した medium として表す。

\[
H =
\sum_{\langle x,y\rangle}
J\!\left(\tau_{\sigma(x)},\tau_{\sigma(y)}\right)
\;+\sum_{\sigma}\lambda_{\tau_\sigma}
\left(V_\sigma-V_{\mathrm{target},\sigma}\right)^2
\;+\lambda_S\left(V_S-V_{\mathrm{target},S}\right)^2
\]

接触項は \(\sigma(x)\ne\sigma(y)\) の辺だけに加える。ランダムに選んだ隣接サイトの ID をコピーし、エネルギー差 \(\Delta H\) に対して

\[
P(\mathrm{accept})=\min\{1,\exp(-\Delta H/T)\}
\]

で受理する。1 Monte Carlo sweep (MCS) は格子サイト数回のコピー試行とする。

### 3.2 単一進行度からの写像

最小 PoC の主駆動は面積要求の交換である。

\[
V_{\mathrm{target},E}(p)=V_{E,0}\{1+\alpha_E s(p)\},
\qquad
V_{\mathrm{target},S}(p)=|\Omega|-N_E V_{\mathrm{target},E}(p)
\]

ここで \(s(p)\) は滑らかな sigmoid または smoothstep とする。固定領域内で E の目標面積が増え、S の目標面積が減るため、明示的な重なりなしに増殖圧と体積排除が生じる。

初期 PoC では \(J\)、\(\lambda\)、\(T\) は全 \(p\) で固定し、1 site 幅の局所 bridge を消して E 細胞を断片化するコピーだけを棄却する。これで相系列が出なければ、次の順で最小限に拡張する。

1. \(J_{ES}(p)\): 上皮–間質界面張力の変化。
2. 分裂: \(V_E\) が閾値を超えた E 細胞を 2 ID に分ける。
3. 極性／方向依存接着: IV_B の分岐を作る。ただし全体方向を固定して平行 stripe を作らない。

### 3.3 観測量

PoC の pit field は E サイトの二値占有場とする。各 \(p\) の場を `pit_goal_metrics.measure_goal_pattern()` に渡し、`classify_morphology()` で既存モデルと比較する。併せて次を記録する。

- E/S 面積率
- 細胞ごとの面積誤差
- E–S 界面長
- MCS あたり受理率
- 総エネルギーと接触項・面積項

初期の E 円盤が I、面積増加中の合体・変形が IIIL/IV_B、E/S がともに連結する領域が IV_V に対応するかを検証する。ラベルは自動分類の結果であり、事前に Type 名を設定しない。

### 3.4 PoC の境界

- S を 1 個の medium とするため、個々の間質細胞の圧排・アポトーシスはまだ表現しない。
- 面積増加は「分裂を含む正味の増殖」の縮約であり、実際の細胞数増加ではない。
- 標準 CPM の格子異方性が出うる。4 近傍接触の PoC 後は、接触エネルギーを 8–20 近傍化して確認する。
- CPM のエネルギー最小化だけで有限長 IIIL が安定する保証はない。過渡状態なら、進行ランプ速度と停止後安定性を分けて評価する。

実装: [`src/pitpattern/models/simulate_pit_cpm.py`](../../src/pitpattern/models/simulate_pit_cpm.py)

### 3.5 初期 PoC 結果

`size=84`, `100 MCS/stage`, `seed=7` の進行ランプでは、E 面積率が 0.106 → 0.598 と増え、孤立領域から連結相へのトポロジー変化は生じた。出力は `artifacts/pit_cpm/contact_sheet.png` と `report.json`。

一方、系列は自動分類上 `IIIL → fail → IV_V → IV_V → IV_V` となるものの、画像上の高 \(p\) は**幅一定の迷路ではなく太い連結 blob** である。したがって `pit_pattern_goal.md` の Must 条件には未達であり、分類器の IV_V はこの PoC では偽陽性と判定する。

この結果から、目標面積交換だけで「孤立→連結」は作れるが、**分岐を進める操作と幅を保つ操作を分離する必要がある**。次の実験は、初期幅を保つ個別 E の分裂と、S チャネルの最小幅制約を優先し、その後に局所極性を検討する。

## 4. off-lattice 2 成分粒子モデル

### 4.1 状態と相互作用

上皮粒子 E と間質粒子 S を位置 \(\mathbf{x}_i\)、種 \(\tau_i\)、必要なら極性 \(\mathbf{q}_i\) で表す。粒子対 \(a,b\in\{E,S\}\) ごとに Morse 型ポテンシャルを置く。

\[
U_{ab}(r)=D_{ab}\left(1-e^{-a_{ab}(r-r_{e,ab})}\right)^2-D_{ab}
\]

\[
\gamma\dot{\mathbf{x}}_i
=-\sum_{j\ne i}\nabla_i U_{\tau_i\tau_j}(r_{ij})
+\mathbf{F}^{\mathrm{polarity}}_i+\boldsymbol{\eta}_i
\]

近距離の急な斥力が体積排除、平衡距離外側の引力が接着を表す。候補関係は \(D_{EE}>D_{ES}\)、E–S は弱接着または実効斥力、S–S は間質の連続性を保つ強さとする。

### 4.2 増殖・圧排・極性

- **E 分裂**: レート \(k_{\mathrm{div}}(p)\) で娘粒子を親の近傍へ挿入する。挿入直後の斥力が増殖圧になる。
- **S 圧排／消失**: 局所圧力または重なりエネルギーが閾値を一定時間超えた S を境界へ移流させる。PoC 後にアポトーシスとして消去する場合は、粒子数と除去理由を記録する。
- **極性**: 低 \(p\) では E が局所リングの接線方向に整列する弱い配向項を持つ。高 \(p\) で配向強度を弱め、増殖圧によるリングの座屈・分岐を許す。

\[
k_{\mathrm{div}}(p)\uparrow,\qquad
k_{\mathrm{death},S}(P,p)\uparrow,\qquad
\kappa_{\mathrm{polarity}}(p)\downarrow
\]

外部操作は \(p\) のみだが、内部係数は固定した滑らかな写像で連動させる。各効果の因果を判別するため、実験では「増殖のみ」「増殖＋S 圧排」「増殖＋S 圧排＋極性破綻」のアブレーションも行う。

### 4.3 pit field への射影

粒子座標から直接 Type を判定せず、内視鏡 top-view に相当する場へ射影する。

1. E 粒子密度または alpha shape から上皮領域を作る。
2. E–S 界面を腺口／溝候補として一定幅で rasterize する。
3. 同じ `pit_goal_metrics` で幅・短管・分岐・迷路を計測する。

射影幅を \(p\) とともに変更してはならない。これにより、力学が生んだ位相変化と描画上の太線化を分離する。

### 4.4 実装順

1. 周期境界、近傍リスト、E/S Morse 力、過減衰緩和。
2. E 分裂と局所圧力の記録。
3. S 圧排／消失。
4. E 極性とリング座屈。
5. rasterize と既存指標への接続。

粒子モデルは CPM より連続座標の力と局所圧力を解釈しやすい一方、pit field への射影と高速な近傍探索が必要である。最小 PoC は極性を後回しにし、1–3 と 5（射影）までを実装する。

実装: [`src/pitpattern/models/simulate_pit_particles.py`](../../src/pitpattern/models/simulate_pit_particles.py)

### 4.5 初期 PoC 結果

`size=64`, `80 step/stage`, `dt=0.03`, `seed=7` では、E 粒子 175→596、S 粒子 656→310、上皮占有率 0.411→0.935 と単調に増殖・圧排が進んだ。出力は `artifacts/pit_particles/contact_sheet.png` と `report.json`。極性は未実装。

形態ラベルは `IIIL → IIIL → IIIL → IV_V → IV_V`。画像上は孤立円盤の肥大・接近・合体であり、高 \(p\) は**幅一定の迷路ではなく太い連結 blob** である。CPM PoC と同じく、分類器の IV_V は偽陽性で、`pit_pattern_goal.md` の Must 条件には未達とする。

指標の pit field は上皮占有場を使った。設計どおり E–S 界面を分類器へ入れると、低 \(p\) の円盤外周がリング網になり、最初から IV_V と誤認される。界面は固定幅で記録するが、主系列の判定には使わない。

この結果から、Morse 力 + E 分裂 + 圧力ゲートの S 消去でも「孤立→連結」は作れる。次は、クラスタをリング（管腔）として初期化し、分裂後も特徴幅を保つこと、および極性項による座屈・分岐を優先する。

## 5. 比較実験

全モデルで seed、初期 pit 間隔、初期幅、視野サイズを可能な範囲で揃え、同じ \(p\) 点列と評価器を使う。

| 問い | CPM | off-lattice |
|---|---|---|
| 増殖圧の表現 | 目標面積誤差 | 分裂後の対粒子斥力 |
| 体積排除 | 1 site = 1 ID | Morse 近距離斥力 |
| 接着 | \(J_{ab}\) | \(D_{ab},a_{ab}\) |
| 間質圧排 | S 目標面積低下 | S の移動／消失 |
| 極性・座屈 | 方向依存接着を追加 | 粒子極性を直接追加 |
| 主な失敗 | 格子異方性、断片化、全面占有 | 時間刻み不安定、射影依存、計算量 |

主評価は [`pit_pattern_goal.md`](../pit_pattern_goal.md) の Must 条件とする。機械的説明がもっともらしくても、幅暴走、IIIL 欠落、平行 stripe、非定常な過渡だけなら未達とする。
