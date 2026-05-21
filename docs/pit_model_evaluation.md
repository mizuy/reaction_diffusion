# Kudo pit pattern を反応拡散モデルで評価する考え方

## 目的

この実験の目的は、pit の幾何模様を外から描くことではなく、反応拡散モデルの数式を変更することで Kudo pit pattern に近い相がどのように生じるかを比較することである。

特に対象にするのは以下の pattern である。

- Type I: 比較的規則的な round pit
- Type III: 小型または大型の round/tubular pit
- Type IV: branching / gyrus-like pattern
- Type VI: 大きさ、形、配列が不整で、複数の pattern regime が混在する状態

Type II と Type VN は、この段階では評価対象から外す。

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

```text
C_t = epsilon (Dc Laplacian(C) + alpha (B - mean(B)) - beta C)
f(C) = f0 + sf C
k(C) = k0 + sk C
```

`C` を固定場ではなく、`B` と結合した遅い変数にする。

期待する効果:

- B pattern が環境を変え、その環境がまた pattern を変える
- 画面内で spot、short stripe、branching、broken stripe が混在する
- VI 不整を「複数の regime が共存する相」として表現しやすくなる

## 評価指標

出力される `metrics.csv` / `metrics.json` には以下の指標が含まれる。

### Type IV 改善を見る指標

`branch_density`

: skeleton 上の分岐点密度。高いほど branching が多い。

`longest_component_fraction`

: skeleton の最大連結成分が全 skeleton に占める割合。高すぎると、長い labyrinth 線が支配的である可能性がある。

`long_line_score`

: 長い線の支配性を branch density で割り引いた指標。Type IV 改善では、この値が下がりつつ `branch_density` が保たれる、または増えることを期待する。

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
- red: branch point
- cyan: endpoint

赤い branch point が増え、かつ長い単一連結線が支配的でなくなるなら、Type IV の改善候補と見る。

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
