# Type IV / VI — 反応飽和・環境場 C（Step 1〜3, 1c）

## この文書の位置づけ

[`pit_model_evaluation.md`](pit_model_evaluation.md) は評価スクリプトの step 一覧・指標・実行方法をまとめた**手順書**である。

本稿は **Step 1（反応飽和）・Step 1b（cubic 減衰）・Step 1c（3 variant）・Step 2（静的環境 C）・Step 3（動的環境 C）** のコンセプト、数式、実測、有望と評価しやすい点、限界をまとめた**設計メモ**である。

後続の Step 4（局所インヒビター H）・Step 5（bistable W）は別メモ:

- [`pit_type_iv_local_inhibitor_concept.md`](pit_type_iv_local_inhibitor_concept.md)
- [`pit_type_iv_bistable_width_concept.md`](pit_type_iv_bistable_width_concept.md)

---

## 共通の土台（Step 0 からの拡張）

Gray-Scott 型の基本形:

```text
A_t = dA ∇²A − R(A,B) + f(1−A)
B_t = dB ∇²B + R(A,B) − (k+f)B
```

Step 1 以降では、おおむね次の **2 系統** を重ねる。

1. **B を直接弱める** — 飽和反応 + cubic / 芯減衰 / A 消費（Step 1, 1b, 1c）
2. **(f, k) を空間的・時間的にゆらす** — 環境場 `C`（Step 2, 3）

いずれも pit の形をマスクで与えず、**式と係数だけ**で morphology を変える。

---

## Step 1: 反応項の飽和

```text
R(A,B) = A B² / (1 + q B²)
```

| コード | `reaction_saturation` (= q) |
|--------|-----------------------------|
| step1 | 例: 1.6 |

`B` が高いほど自己増殖が頭打ちになり、太く長い B-rich stripe の過剰安定を弱める。

---

## Step 1b: cubic damping

```text
B_t = … − μ B³
```

| コード | `cubic_damping` (= μ) |
|--------|------------------------|
| step1b | 例: 0.018 |

高濃度 B にペナルティをかけ、stripe の芯を抑える。Step 1 と併用するのが Step 1c 系の前提。

**Step 1 + 1b の位置づけ**: B の「強さ」を下げる**対症療法**。長さスケールや幅の固定ではない。

---

## Step 1c: 飽和・cubic の追い込み（3 variant）

`step1b`（`feed=0.033`, `k=0.056`, `reaction_saturation=1.6`, `cubic_damping=0.018`）を土台に、さらに 3 通りを比較した。

### step1c_a — `step1c_a_strong_cubic`

**変更**: cubic を強める（`cubic_damping=0.028`）。

**意図**: 後期の stripe 粗化・長い labyrinth の支配を抑える。

**力学**: μ を上げると高 B 全体が抑えられ、線は短く・細くなるが、**分岐も一緒に消えやすい**。

### step1c_b — `step1c_b_a_depletion`（失敗例）

**変更**: A 方程式に B-rich での追加消費。

```text
A 消費 += η A B        （η = a_depletion, 例 0.25）
```

**意図**: stripe 内部で基質 A を奪い、芯を弱める。

**結果**: 基質を奪いすぎて **B が全滅**（`binary_coverage=0`）。パラメータ探索の失敗例として残す。

### step1c_c — `step1c_c_uniform_core`（1c で最もバランスが良い）

**変更**: **平坦な高 B 芯だけ** に減衰（先端は残す）。

```text
core(x,y) = (B > b_min) ∧ (|∇B| < grad_max)
B 減衰 += δ_core · core · B
```

| コード | 意味 | 例 |
|--------|------|-----|
| `uniform_core_decay` | δ_core | 0.12 |
| `uniform_core_b_min` | 芯とみなす B の下限 | 0.55 |
| `uniform_core_grad_max` | 「平坦」とみなす勾配上限 | 0.035 |

**意図**: 1c_a のように全体を弱めるのではなく、**太い芯だけ**を殺し、勾配の大きい **先端・界面** は反応を続けられるようにする。

**読み方**: 「背後の均一な芯は死ぬが、先端は生きる」という、後の refractory 場アイデアに近い**空間的選択的減衰**（ただし時間記憶はない）。

---

## Step 2: 静的な環境場 C

初期化時に滑らかなノイズ `C(x,y)` を作り、**固定**したまま `f,k` を空間変動させる。

```text
f(x,y) = f₀ + s_f C(x,y)
k(x,y) = k₀ + s_k C(x,y)
```

| コード | 役割 |
|--------|------|
| `static_env_scale` | 初期 C の振幅 |
| `feed_env_sensitivity` | s_f |
| `k_env_sensitivity` | s_k |

`C` は pit 形状そのものではなく、粘膜・栄養などの**ゆるい組織ムラ**の代理。

**限界**: C が固定なので、B pattern が後から環境を書き換えるフィードバックはない。Step 3 の前段階。

---

## Step 3: 動的な環境場 C — `step3_pit_vi_dynamic_environment`

### 数式

```text
f(x,y) = f₀ + s_f C
k(x,y) = k₀ + s_k C

C_t = ε ( D_C ∇²C + α (B − ⟨B⟩) − β_C C )
```

コード上（`advance_reaction_diffusion`）:

```text
next_c = c + env_rate * (
    env_diffusion * lap(c)
    + env_source * (b - mean(b))
    - env_decay * c
)
```

| 記号 | フィールド | 例（step3） | 役割 |
|------|------------|-------------|------|
| ε | `env_rate` | 0.035 | C の更新の遅さ |
| D_C | `env_diffusion` | 0.20 | C の空間平滑化 |
| α | `env_source` | 0.70 | B が平均より高い所で C を増やす |
| β_C | `env_decay` | 0.08 | C の減衰 |
| s_f | `feed_env_sensitivity` | 0.0045 | C → feed |
| s_k | `k_env_sensitivity` | −0.0025 | C → k |

初期 `C` は Step 2 と同様、`static_env_scale` で滑らかノイズ（例 0.8）を与えたうえで、以降は **B と結合して進化**する。

### 力学のイメージ

```text
  B が増える領域
       |
       v
  C がそこで上がる（遅い）
       |
       v
  局所の f, k がずれる → 反応拡散の regime が変わる
       |
       v
  B pattern がまた変わる → …
```

- **時間スケール**: C は `env_rate` で遅く更新（毎 step の A/B より遅い「環境」）。
- **空間スケール**: `env_diffusion` で C は滑らか。
- **狙い**: 画面内に **spot / 短い stripe / 分岐 / 断片** が混在する **Type VI 不整**（複数 regime の共存）。

Step 4 の **H**（速い局所抑制）とは別物。Step 3 は **「その場の反応条件を B が書き換える」遅いループ**。

### step3 のその他の係数

baseline よりやや高めの `feed=0.036`, `k=0.058`。飽和・cubic は 1b より弱め（`reaction_saturation=1.2`, `cubic_damping=0.012`）— 環境 C を主役にしつつ B の過剰抑制を避けるチューニング。

---

## 実測比較（size=160, steps=3500, seed=7）

`artifacts/pit_step_eval/metrics.json` より。被覆 `binary_coverage` はおおむね ~0.30 前後（1c_b を除く）。

| case | longest | branch | br/comp | vi_irreg | 解釈 |
|------|---------|--------|---------|----------|------|
| step0 baseline | 0.83 | 86 | 6.1 | 3.34 | 巨大迷路 |
| step1c_a strong_cubic | **0.08** | 4 | 0.11 | 0.60 | 迷路は壊れるが **枝ほぼ消失**（孤立片寄り） |
| step1c_b a_depletion | 0 | 0 | — | 0 | **全滅** |
| **step1c_c uniform_core** | 0.15 | 14 | 0.42 | 0.83 | 迷路分断 + **分岐を少し保持** |
| step2 static C | （手順書・metrics 参照） | | | | 空間ムラのみ |
| **step3 dynamic C** | **0.21** | **28** | **1.4** | 1.07 | 迷路↓ + **分岐維持** + **不整さ** |
| step5 W（参考） | 0.33 | 26 | 1.4 | 1.63 | 幅固定系の比較対象 |

### なぜ step1c_c / step3 が「有望」に見えやすいか

**step1c_c**

- 1c_a と違い、**3 股分岐（branch）がゼロにならない**（`br/comp ≈ 0.42`）。
- `longest` を baseline 0.83 から 0.15 へ下げ、**一枚の巨大 skeleton の支配を弱める**。
- 「芯だけ弱く、先端は残す」設計が、数値上も「短い + 少し枝あり」に寄っている。

**step3**

- `longest ≈ 0.21` で迷路支配を抑えつつ、`branch ≈ 28`, `br/comp ≈ 1.4` は **step5 W 単体と同オーダー**。
- `vi_irregularity_score` が baseline より低く落ち着く一方、**成分サイズのばらつき**（`component_area_cv`）は出ており、**複数 morphology の共存**（VI 寄り）の候補。
- `c_std ≈ 0.93` と C が場で動いていることと整合。

### 限界（後続 step との関係）

| 課題 | Step 1c / 3 の限界 | 後続での補い |
|------|-------------------|--------------|
| 長い迷路 | 弱められるが、1c_a は枝も死ぬ | H（間隔）、W（幅） |
| 巨大迷路 ↔ 孤立片の往復 | 1c 系で顕著 | W + 厳密分岐指標 |
| 幅が (f,k) に吸収される | 未解決 | Step 5 bistable |
| 速い側方抑制・先端失速 | Step 3 は遅い C のみ | Step 4 H |

Step 1c / 3 は **有望な中間地点** だが、「伸び続ける力学に内在する長さスケールを足す」（H）や「幅を固定する」（W）とは **直交するノブ** として再検討する価値がある（例: **step3 + 弱い W**、**step1c_c + 弱い H**）。

---

## Step 2 と Step 3 の対比

| | Step 2 静的 C | Step 3 動的 C |
|--|----------------|----------------|
| C の時間発展 | なし（初期ノイズで固定） | あり（B に追従） |
| 結合 | f,k の空間ムラ | **B → C → (f,k) → B** |
| 主な狙い | Type IV の線を局所で変える | **VI 不整**（regime 共存） |
| case 名 | `step2_pit_iv_static_environment` | `step3_pit_vi_dynamic_environment` |

---

## 画像・評価の見方

各 case の出力:

```text
artifacts/pit_step_eval/<case_name>/final_fields.png
artifacts/pit_step_eval/<case_name>/final_skeleton.png
```

- **step1c_c**: 均一な長い1本ではなく、複数の曲線・断片；skeleton で赤（3-arm fork）が散らばっているか。
- **step3**: fields で spot と stripe が混在；`C` チャンネル（緑）が B 分布と相関して変化しているか（`fields_*.png` の時間列）。

指標の定義は [`pit_model_evaluation.md`](pit_model_evaluation.md) の「評価指標」節。

---

## Step 1c と Type IIIL / Type IV — なぜ IV になりにくいか

### 相の違い（ざっくり）

| タイプ | Gray-Scott 上のイメージ | 典型 (f, k) 目安 | step1c 系の指標の傾向 |
|--------|-------------------------|------------------|----------------------|
| **III / IIIL** | spot・太い tubular / round pit | f やや大 (例 0.045, k 0.066) | `a_mean` 高 (~0.6–0.7)、`longest` 低、成分多数・面積 CV 中 |
| **IV** | labyrinth・分岐する stripe | f やや小 (例 0.033, k 0.056) | `branch`・`br/comp` 中〜高、`longest` 中 (0.2–0.5) |

step1c（特に **1c_c**）は **飽和 q≈1.6 + 芯減衰** で「高 B の塊を丸く・短く保つ」方向に強く働く。これは **IIIL に近い** 一方、**stripe がつながって分岐する IV の相** から遠ざける。

`feed` だけを動かすと、

```text
A_t に +f(1−A)   →  f↑ で A が補充されやすい（a_mean↑）
B_t に −(k+f)B   →  f↑ で B が消えやすい（b_mean↓、全滅リスク）
```

と **A と B で効く符号が逆** なので、「総量バランスが崩れる」と感じやすい。特に飽和・芯減衰がかかっていると、**f の許容幅は ±0.001 程度** と狭い。

### 目標 morphology：楔形 IIIL 分岐 vs 迷路 IV（観察の整理）

臨床イメージに近いのは、おそらく次の **中間形態** である。

```text
  望ましい:  短い丸い pit（IIIL）が 2–3 本、楔形（Y 字）で分岐
            各枝は有限長。全体は「木」や「楔形の集まり」

  避けたい:  枝がだんだん伸び、つながり、画面を横断する labyrinth（经典 IV）
```

`reaction_saturation` を下げると **分岐指標は上がる** が、見た目はしばしば次の順で変化する。

1. **q 高（1.6）** — 丸い IIIL 断片、分岐ほぼなし  
2. **q 中（1.0–1.2）** — **短い太い pit が Y 字・楔形に分岐**（いまの「手応り」）  
3. **時間経過 + q 低** — pit 同士が **細い stripe で橋渡し** → 橋が伸びる → **迷路化**

つまり「分岐が出た」ことと「Type IV の labyrinth ではない」ことは両立する。  
指標の `branch` は楔形の 3 股でも赤くなるが、**`longest` が時間とともに上がる** のは **連結・粗化（coarsening）** のサインである。

```text
  単一の大域 q では両立しにくい

  q 大  →  spot 安定、分岐弱い、短い（IIIL ✓, 分岐 ✗）
  q 小  →  stripe 反応強い、つながる、伸びる（分岐 △, 迷路 ✓）

  楔形 IIIL 分岐 = 「局所的に q が小さい（先端）」+ 「大域的に q が大きい（芯・背後）」
                    + 「橋が伸び続けるのを止める項」
```

**結論**: `feed` / `q` のスライダー調整だけでは、**「短い楔形分岐の木」を安定固定**するのは難しい。**成長の場所**と**成長の持続時間**を分離する式の追加が必要、というのが現在の診断である。

### exp3 での調整の優先順位（パラメータのみの限界）

1. **`reaction_saturation`（react sat）を下げる** — 分岐は出るが、上記のとおり **楔形 → 迷路** になりやすい。  
   - q=1.6: `branch≈0`, `longest≈0.12`（IIIL、分岐弱）  
   - q=1.2: `branch≈8`, `longest≈0.34`（中間）  
   - q=1.0: `branch≈17`, `longest≈0.49`（連結・伸長寄り）  
   **中間 q で「楔形のスナップショット」を取る**のは有効だが、**長時間は迷路化を止められない**。

2. **`uniform_core_decay`（core δ）を弱める** — q を下げたうえで **0.12 → 0.06–0.08**。芯の丸さは残し、stripe の連結・分岐を戻しやすい。

3. **`cubic_damping`（μ）** — 1c_c では 0.018 のままか **0.012** まで下げる。上げすぎると 1c_a 同様に枝が消える。

4. **`feed` / `k`** — **同時に大きく動かさない**。  
   - `feed` を **+0.001〜+0.002**（0.034–0.035）だけ上げると分岐が増えることがあるが、`a_mean` が 0.65 超なら戻す。  
   - `k` を下げすぎ（0.054）すると **巨大迷路**（`longest≈0.84`）に戻りやすい。  
   - `f` と `k` を両方上げる（0.034, 0.057）と B が死にやすい（失敗パターン）。

5. **`d_b` を少し上げる**（0.5→0.55）— stripe 幅・連結に効くが、効果は q より小さい。

**触らない方がよい方向**: reference III 寄りの (f,k)=(0.045, 0.066)、強い core+強い q のまま feed だけ大きく動かす。

### ライブ指標の目安（exp3 右上）

| 観察 | IIIL 寄り | IV 寄り（1c 土台） |
|------|-----------|-------------------|
| `a_mean` | > 0.62 | 0.45–0.58 |
| `b_mean` | 0.10–0.14 | 0.14–0.20 |
| `cov` | ~0.30 | ~0.30（全滅 0 / 過密に注意） |
| `longest` | < 0.15 | 0.25–0.45 |
| `branch` / `br/comp` | ≈ 0 | > 0.5 |

---

## モデル修正の提案（楔形 IIIL 分岐を固定し、迷路化を止める）

パラメータ調整（上節）より **次の式レベルの分離** を優先する。

### 優先度 A — 成長を「先端だけ」に限定（飽和の空間化）

いまの `uniform_core_decay` は「芯を弱める」が、**先端の反応は大域 q で抑えたまま** にもできる。逆に **先端だけ q を下げる** と楔形分岐が出る。

```text
q_eff(x) = q_core   if core(B, ∇B)     # 平坦な高 B（丸い IIIL の体）
         = q_tip    otherwise         # 界面・先端（q_tip < q_core）

R = A B² / (1 + q_eff B²)
```

- **q_core 大（1.4–1.6）** — 芯は spot のまま、横に伸びない  
- **q_tip 小（0.2–0.8）** — 先端だけ成長・分岐  
- 既存の `uniform_core_mask` を流用できる → **実装コスト小、第一候補**

### 優先度 A' — 背後だけ死ぬ不応場 R（refractory）

楔形は「先端は生き、背後は死ぬ」構造。時間積分で実装する。

```text
R_t = R_0 B − R           # B が滞在した場所に蓄積
B 減衰 += α_R · R · B     # 古い芯・古い corridor だけ強く kill
```

- stripe が **有限長の「脚」** で止まる  
- H（拡散抑制）とは別：**時間メモリ** で長さ上限を付ける  
- Step 4 の「不応場 W」アイデアの実装版（[`pit_type_iv_local_inhibitor_concept.md`](pit_type_iv_local_inhibitor_concept.md) 参照）

### 優先度 B — 橋渡し stripe の粗化を抑える（反 coarsening）

楔形のあと **pit 間をつなぐ細い B** が伸び、经典 labyrinth になるのは **長波長 coarsening** に近い。

```text
B_t += … − ν ∇⁴ B        # または −κ ∇²(∇²B)
```

- 長い1本の stripe が伸びるのを嫌う（界面エネルギー的ペナルティ）  
- spot / 短い棒は残しやすい  
- 係数 ν は小さく始める（数値安定に注意）

### 優先度 B' — 局所インヒビター H（間隔固定、弱め）

H は強すぎると断片化、弱すぎると迷路。楔形を安定させるには **1c_c + ごく弱い H** で「pit 同士の最短距離」を決め、**橋が長く伸びる前に隣 pit の H 雲と衝突** させる。

- 目標: `longest` を抑えつつ `branch` を維持（step5b の失敗は H が強すぎた例）  
- **γ, ρ を step4 の半分以下** からスイープ

### 優先度 C — 幅固定（bistable W）+ 1c_c

[`pit_type_iv_bistable_width_concept.md`](pit_type_iv_bistable_width_concept.md) の W は **太さ** を固定し、伸長を topology に流す。  
IIIL の「丸い体」には **W + 大 q_core** の組み合わせが向く可能性がある（伸長で太さが変わらず、分岐だけ動く）。

### 優先度 D — 動的 C で regime 共存（step3 拡張）

局所によって spot 相（III）と stripe 相（IV）を混ぜる。  
**全場が一つの相に引きずられて迷路化**するのを避け、**III 域は pit のまま**、IV 域だけ繋がる、といった **パッチワーク** を狙う（VI 不整）。

### Step 1d（実装済み）— 空間 q と不応 R

評価スイート `step1d_*` と `exp3` プリセットで、上記 **優先度 A / A'** をコード化した。

**空間飽和**（`reaction_saturation` = q_core、`reaction_saturation_tip` = q_tip）:

```text
q_eff = q_core  if uniform_core_mask(B)   # 平坦な高 B 芯（IIIL の体）
      = q_tip   otherwise                 # 界面・先端
R = A B² / (1 + q_eff B²)
```

**不応場 R**（`step1d_b` のみ。kill は **core マスク内** に限定）:

```text
R_t += refractory_rate · B − refractory_decay · R
B_t += … − refractory_strength · R · B · core_mask
```

| case | q_core / q_tip | その他 | branch | longest | a_mean | 所見 |
|------|----------------|--------|--------|---------|--------|------|
| step1c_c | 1.6 / — | core δ=0.12 | ≈14 | ≈0.15 | ≈0.64 | IIIL 寄り、分岐弱 |
| **step1d** | **1.5 / 0.5** | core δ=0.10 | **≈48** | **≈0.35** | **≈0.52** | 楔形・分岐↑、迷路寄りも |
| step1d_b | 1.5 / 0.5 | R 弱（芯のみ kill） | ≈39 | ≈0.35 | ≈0.52 | 分岐やや↓、longest は 1d と同程度 |
| step0 | — | baseline | ≈86 | ≈0.83 | — | 巨大迷路 |

`q_tip` のスイープ（seed=7, 3500 step）では **0.50** が `longest` 最小（≈0.35）。**0.45** は分岐は多いが `longest≈0.91` で baseline 級の連結。**0.55** は `longest≈0.49` で中間。プリセットは **q_core=1.5, q_tip=0.5** を採用。

**限界**: 空間 q だけでも `longest` は step1c_c より大きく、時間で pit 間が橋渡しされ迷路化しやすい。不応 R を全場にかけると pattern 全滅 → **芯限定 kill** が必須。楔形木の KPI（`longest` の時間微分 ≈ 0）は、まだ step1c_c より満たしにくい。

### おすすめの実装順

| 順 | 案 | 狙い | 状態 |
|----|-----|------|------|
| 1 | **空間 q_eff**（core / tip） | 楔形 IIIL 分岐そのもの | **Step 1d 実装済** |
| 2 | **不応 R** または **弱 H** | 脚の最大長・迷路化の抑制 | **R: step1d_b 実装済** / H は step4 |
| 3 | **∇⁴B**（弱い） | 橋の伸長・粗化の抑制 |
| 4 | **1c_c + W** | feed 調整で形が壊れる問題の回避 |

**パラメータのみ**の `step1c_c_iv` プリセットは、**中間スナップショット用**には残す価値があるが、**3500 step 後も楔形を保つ**用途には上記 1–2 が必要、と整理する。

### 評価の見直し（楔形 vs 迷路）

| 指標 | 楔形 IIIL 木 | 迷路 IV |
|------|----------------|---------|
| `longest` | **低いまま時間一定**（< 0.2） | 時間とともに上昇しやすい |
| `mean_seg` | 短い（ピット径程度） | 長くなる |
| `branch` / `br/comp` | 中程度（3 股） | 高いが「1 枚の網」 |
| 目視 | 丸い端、Y 字、楔形 | serpentine、画面横断 |

**成功条件**を「branch が多い」だけにしない。**`longest` の時間微分 ≈ 0** かつ **成分が多数のまま** を楔形木の KPI に追加するのがよい。

---

## 実装との対応

| 項目 | 場所 |
|------|------|
| 飽和反応 `R` | `reaction_term_with_config()`（空間 q_eff 対応） |
| cubic / uniform_core / a_depletion | `advance_reaction_diffusion()` |
| 不応場 R | `advance_reaction_diffusion()`（`grid_r`、core 限定 kill） |
| 静的・動的 C | `build_static_environment()`, `advance_reaction_diffusion()` の `dynamic_env` ブロック |
| effective f,k | `effective_parameters()` |
| suite 定義 | `build_suite()` → `step1c_*`, **`step1d_*`**, `step2_*`, `step3_*` |
| 一括評価 | `python3 src/evaluate_pit_steps.py --output artifacts/pit_step_eval` |

対話シミュレーション: `uv run python src/exp3.py` または `make exp3`（プリセット: step1b / step1c_a,b,c / **step1d / step1d_b** / step3）。3×2 レイアウト（A/B | skeleton+metrics | 環境 C または不応 R）、プリセットごとに関連スライダーのみ表示。`exp.py` は全パラメータ統合版。

---

## まとめ

- **Step 1 / 1b**: 飽和 + `B³` で B-rich stripe を弱める対症療法。
- **Step 1c**: 3 通りの追い込みのうち **uniform_core（1c_c）** が、迷路分断と分岐保持のバランスが最も良い。**strong_cubic（1c_a）** は短いが枝なし、**a_depletion（1c_b）** は全滅。
- **Step 2**: 固定の環境ムラで f,k を空間変動。
- **Step 3**: **B と結合して進化する遅い環境 C** で、迷路支配を下げつつ分岐・不整さを残しやすい。**Type VI 寄り**の候補。
- **再検討の方向**: 楔形 IIIL 分岐を主目標にし、**空間飽和（q_core / q_tip）** と **不応 R または弱 H** で迷路化を止める。単一の `reaction_saturation` スライダーは中間相の探索用と割り切る。
