# Kudo pit pattern を反応拡散モデルで評価する考え方

> pit pattern（工藤分類）の医学的背景・各 Type の定義・反応拡散モデルとの対応は `docs/pit_pattern_literature.md` にまとめている。

## 目的

この実験の目的は、pit の幾何模様を外から描くことではなく、反応拡散モデルの数式を変更することで Kudo pit pattern に近い相がどのように生じるかを比較することである。

特に対象にするのは以下の pattern である。

- Type I: 比較的規則的な round pit
- Type III: 小型または大型の round/tubular pit
- Type IV: branching / gyrus-like pattern
- Type VI: 大きさ、形、配列が不整で、複数の pattern regime が混在する状態

Type II と Type VN は、この段階では評価対象から外す。

設計の背景・Type IV の力学診断は次の設計メモを参照する。

- 反応飽和・環境場 **C**（Step 1〜3, 1c）: [`pit_type_iv_saturation_environment_concept.md`](pit_type_iv_saturation_environment_concept.md)
- 局所インヒビター場 **H**（Step 4）: [`pit_type_iv_local_inhibitor_concept.md`](pit_type_iv_local_inhibitor_concept.md)
- bistable plateau **W**（Step 5・幅の固定）: [`pit_type_iv_bistable_width_concept.md`](pit_type_iv_bistable_width_concept.md)

## 現在の基本モデル

既存コードは Gray-Scott 型の反応拡散モデルとして読める。

```text
A_t = dA Laplacian(A) - A B^2 + f(1 - A)
B_t = dB Laplacian(B) + A B^2 - (k + f)B
```

ここで `A` と `B` は反応物質、`f` は feed、`k` は kill rate である。

既存モデルでは `f` と `k` の選び方によって Type I、Type III、Type IV に近い pattern がある程度出る。ただし Type IV では長い labyrinth 状の線が安定しやすく、実際の pit pattern に期待される短い branching が不足しやすい。

## 問題意識

Type IV の課題は、線状構造が「伸び続ける」方向に安定しすぎることだと考える。

そのため、直接線を切ったり枝を描いたりするのではなく、以下を数式側で起こしたい。

- B-rich stripe の過剰な安定化を弱める
- stripe の先端や途中で分裂、停止、枝分かれが起きやすくする
- 空間的に少し異なる pattern regime が共存できるようにする

VI 不整については、さまざまな pit を幾何的に混ぜるのではなく、局所的に異なる反応拡散 regime が同じ場に共存している状態として扱う。

## Step ごとの仮説

評価スクリプト `src/evaluate_pit_steps.py` は、同じ初期ゆらぎから以下の variant を走らせる。

### Step 0: Type IV baseline

```text
R(A, B) = A B^2
```

元の Gray-Scott モデルを Type IV 候補パラメータで走らせる。

これは比較の基準であり、長い線がどれだけ支配的になるかを見る。

### Step 1: 反応項の飽和

```text
R(A, B) = A B^2 / (1 + q B^2)
```

`B` が高濃度になった場所で自己増殖が強くなりすぎないようにする。

期待する効果:

- 太く長い stripe の安定性を下げる
- 線が局所的に切れやすくなる
- 先端や境界で分岐が生じやすくなる

### Step 1b: cubic damping

```text
B_t = dB Laplacian(B) + R(A, B) - (k + f)B - mu B^3
```

高濃度の `B` をさらに抑える非線形減衰を入れる。

期待する効果:

- B-rich stripe の芯が過剰に安定することを防ぐ
- 長い線より短い枝や断片が残りやすくなる
- Type IV の branching を増やす候補になる

### Step 1c: 飽和・cubic の追い込み

`step1b` の派生として、B 芯をさらに抑える 3 variant（`step1c_a` / `step1c_b` / `step1c_c`）を比較した。数式・実測・**step1c_c / step3 が有望に見える理由**は [`pit_type_iv_saturation_environment_concept.md`](pit_type_iv_saturation_environment_concept.md) を参照。

要約:

- `step1c_a_strong_cubic`: cubic 強化 → 迷路は壊れるが**枝も消えやすい**（孤立片寄り）。
- `step1c_b_a_depletion`: B-rich で A 消費 → **全滅**（失敗例）。
- `step1c_c_uniform_core`: **平坦な高 B 芯だけ**減衰、先端は残す → 1c 系でバランスが最も良い候補。

教訓: 「B 芯を弱める」だけでは巨大迷路と孤立片の往復になりやすい（後続の H / W で補う）。

### Step 1d: 空間飽和（q_core / q_tip）と不応 R

楔形 IIIL 分岐を「先端だけ低 q・芯は高 q」に分離し、迷路化を **不応 R**（古い芯の kill、core マスク内のみ）で抑える段階。数式・スイープ・限界は [`pit_type_iv_saturation_environment_concept.md`](pit_type_iv_saturation_environment_concept.md) の **Step 1d** 節を参照。

```text
q_eff = q_core  on flat high-B core
      = q_tip   on tips / interfaces
R = A B² / (1 + q_eff B²)

R_t += rate·B − decay·R
B_t += … − strength·R·B·core_mask   # step1d_b のみ
```

- `step1d_pit_iiil_spatial_sat`: q_core=1.5, q_tip=0.5, uniform_core δ=0.10 — **分岐↑**（baseline より `branch` 大、`longest` は 1c_c より大きい）。
- `step1d_b_pit_iiil_spatial_refractory`: 上記 + 弱い不応 R — 全場 kill は全滅のため **芯限定**。

対話: `make exp3`（プリセット `step1d_*`、スライダー: q_core / q_tip / R）。

### Step 1e: 極性場 P + 異方性 B 拡散

成長方向 **P** を ∇B に整列させ、**B の拡散を P 方向（強）と直交方向（弱）に分離**する prototype。横方向 coarsening（pit 同士の橋渡し）を抑える狙い。

```text
P_t = d_p Lap(P) + α ∇B − δ P
D_B = d_b n n^T + d_b_across (I − n n^T)   （A は等方のまま）
```

- `step1e_pit_iv_polarity_aniso`: baseline + P/aniso
- `step1e_b`: step1c_c + P 場（**aniso mix=0** — 1c と同指標、P 可視化用）
- `step1e_c`: step1d + 弱 aniso（mix≈0.30）

対話: `make exp4`（第4ペイン: 極性 P の RGB 表示）。スライダー **aniso mix** で等方↔異方のブレンド。

### Step 2: 静的な環境場 C

```text
f(x, y) = f0 + sf C(x, y)
k(x, y) = k0 + sk C(x, y)
```

`C` は滑らかなノイズ場であり、pit の形を直接与えるものではない。粘膜環境、栄養、増殖能などのゆるい空間的不均一性として扱う。

期待する効果:

- 完全一様な場で長い線が伸び続けることを抑える
- 局所ごとに branch の発生や停止が変わる
- Type IV の線が短く複雑になる

### Step 3: 動的な環境場 C

`step3_pit_vi_dynamic_environment`: `C` を **B と結合した遅い変数** にし、`f,k` を局所で書き換える（**B → C → (f,k) → B**）。Type VI 不整（複数 regime 共存）の候補。詳細・実測表は [`pit_type_iv_saturation_environment_concept.md`](pit_type_iv_saturation_environment_concept.md)。

```text
C_t = epsilon (Dc Laplacian(C) + alpha (B - mean(B)) - beta C)
f = f0 + sf C
k = k0 + sk C
```

同一 IC では `longest ≈ 0.21`、`branch ≈ 28`、`br/comp ≈ 1.4` と、迷路支配を下げつつ分岐を残しやすい（step5 W と同オーダー）。

### Step 4: 速い局所インヒビター場 H

Step 1/1b/1c が「B を弱める」対症療法だったのに対し、Step 4 は Type IV が長い迷路になる根本原因(=成長を止める長さスケールが無いこと)を直接補う。

```text
B_t = dB Laplacian(B) + R(A, B) - (k + f)B - gamma B H
H_t = dH Laplacian(H) + rho B - delta H        ( dH >> dB )
```

`H` は `B` から生成され、速く広く拡散して `B` を抑える側方抑制場である。Step 3 の `C` と違い、遅い大域結合ではなく**速い局所抑制**として入れる点が要点。

期待する効果:

- 線同士・先端が H 雲を介して反発し、間隔(=長さスケール)が決まる
- 先端が自分の影に近づくと失速し、伸び続けずに分裂・停止する
- broken stripe と spot が共存し、VI 不整も出やすくなる

`step4b` は H に弱い cubic damping を併用し、短い樹状分岐をさらに狙う variant。

### Step 5: 幅・間隔・被覆の分離（bistable plateau / coverage feedback）

Step 0〜4 で繰り返し現れた問題は、`(f, k)` が **stripe の幅・振幅・被覆率・波長・morphology を同時に決めてしまう**ことだった。幅が「やわらかいモード」として摂動を吸収するため、`(f, k)` を動かしても分岐ではなく太さの変化に逃げてしまう。

Step 5 は「幅・間隔・被覆を別々のメカニズムで固定し、変動を topology(分岐・伸長)へ流す」方針をとる。提案 **W** の詳細（問題の診断・数式・パラメータ・実測）は [`pit_type_iv_bistable_width_concept.md`](pit_type_iv_bistable_width_concept.md) を参照。

提案 W（bistable plateau, `step5_pit_iv_bistable_width`）:

```text
B_t = dB Laplacian(B) + R(A, B) - (k + f)B + lambda B (1 - B) (B - beta)
```

cubic 項で `B` を `0` / `1` の双安定にし、「on」濃度(=stripe の振幅・幅)を `(f, k)` から切り離して `lambda`、`dB`、`beta` で決める。これにより太さが固定され、`(f, k)` の変動が morphology に向かう。

提案 M（global coverage feedback, `step5c` で併用）:

```text
f_eff = f + kappa (phi* - mean(B))
```

大域の被覆率(平均 B)を目標 `phi*` に保つよう feed を弱く調整し、被覆率を固定する。

variant は次の 3 つ。

- `step5_pit_iv_bistable_width`: 提案 W のみ。幅を固定。
- `step5b_pit_iv_bistable_inhibitor`: 提案 W + Step 4 の H。幅と間隔を固定。
- `step5c_pit_iv_three_channel`: 提案 W + H + M。幅・間隔・被覆をすべて固定し、topology だけを自由度として残す。

実測（size=160, steps=3500, default seed）では、被覆率を ~0.31 に保ったまま:

- baseline: `longest 0.83`, `branch 86`, `br/comp 6.1`（巨大迷路）
- step4 H 単体: `longest 0.11`, `branch 0`（迷路は崩れるが厳密 3 股は消え断片化）
- **step5 W 単体: `longest 0.33`, `branch 26`, `br/comp 1.4`（迷路を分断しつつ 3 股分岐を保持）**
- step5b W+H: `longest 0.18`, `branch 0`（H が分岐を抑えすぎる）
- step5c W+H+M: `longest 0.23`, `branch 20`, `br/comp 0.8`（W と H の中間）

要点は、**幅を固定する提案 W だけが、被覆率一定のまま「迷路の分断」と「3 股分岐の保持」を両立する**こと。これは「太さが A/B 比を吸収して分岐を妨げる」という見立てを定量的に裏付ける。H を強く足すと再び分岐が潰れるため、間隔制御は控えめにするのがよい。

## 評価指標

出力される `metrics.csv` / `metrics.json` には以下の指標が含まれる。

### Type IV 改善を見る指標

`branch_density`

: skeleton 上の **3 股分岐点** の密度。判定手順は次のとおり。(1) spur 除去。(2) 次数 ≥ 3 の像素を 8 連結で **junction クラスタ** にまとめる。(3) クラスタの外側へ skeleton に沿って各廊下（corridor）を追跡し、次の junction または端点までの長さを枝長とする。(4) **異なる 3 本の廊下**があり、それぞれが `max(4, size/32)` ピクセル以上のときだけ fork と数え、該当クラスタの像素を赤で示す。1 ピクセル近傍の次数だけでは判定しない。

`longest_component_fraction`

: skeleton の最大連結成分が全 skeleton に占める割合。高すぎると、長い labyrinth 線が支配的である可能性がある。

`long_line_score`

: 長い線の支配性を branch density で割り引いた指標。Type IV 改善では、この値が下がりつつ `branch_density` が保たれる、または増えることを期待する。

`mean_segment_length`

: skeleton から分岐点を除いた区間(セグメント)の平均長。「線が分岐・停止するまでにどれだけ伸びるか」を直接測る。Type IV 改善ではこの値が下がることを期待する。

`branch_per_component`

: 連結成分あたりの分岐点数。短い**樹状**構造(高い=良)と、孤立ダッシュ(0 に近い=悪)や 1 枚の巨大迷路を区別する。`step1c_a` のように線が砕けて枝も消えた状態を検出できる。

### VI 不整を見る指標

`component_area_cv`

: 二値化された pattern の連結成分面積の変動係数。pit サイズや構造サイズのばらつきを見る。

`local_density_std`

: 画面を tile に分けたときの pattern 密度の標準偏差。局所的な密度むらを見る。

`local_density_entropy`

: 局所密度分布の entropy。複数の密度 regime が混じるほど上がりやすい。

`vi_irregularity_score`

: `component_area_cv`、`local_density_std`、`local_density_entropy` を組み合わせた暫定指標。VI らしさの最終判定ではなく、比較のための目安である。

## 画像出力の読み方

各 case には以下が保存される。

```text
final_fields.png
final_skeleton.png
fields_*.png
```

`final_fields.png` は `A`、`B`、必要に応じて `C` を可視化したもの。

`final_skeleton.png` は `B` pattern を二値化して skeleton 化した評価用画像である。

- white: skeleton
- red: 3-way branch（3 本の枝が十分な長さで伸びている junction）
- cyan: endpoint

赤い 3-way branch が増え、かつ長い単一連結線が支配的でなくなるなら、Type IV の改善候補と見る。

## 注意点

この評価は医学的な Kudo classification を直接判定するものではない。あくまで、反応拡散モデルの式変更が pattern morphology に与える影響を比較するための実験基盤である。

また、二値化や skeleton 化の閾値は評価指標に影響する。したがって、単一の数値で判断せず、`contact_sheet.png` と各 step の画像を見ながら指標を解釈する必要がある。

## 実行例

```bash
python3 src/evaluate_pit_steps.py --output artifacts/pit_step_eval
```

軽い確認:

```bash
python3 src/evaluate_pit_steps.py \
  --output artifacts/pit_step_eval_smoke \
  --suite steps \
  --size 96 \
  --steps 200 \
  --snapshot-count 3
```
