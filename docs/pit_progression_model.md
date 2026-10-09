# Pit progression model — 設計メモ

本稿は [`pit_pattern_goal.md`](pit_pattern_goal.md) の I → IIIL → IV_B → IV_V 連続遷移を実装するための**新規モデル**の設計記録である。既存 Gray-Scott Step 合成ではなく、有限波長場モデル + 単一進行度 `p` から出発した。

## 1. 外部パラメータ

| 記号 | 意味 |
|------|------|
| `p ∈ [0, 1]` | 唯一の外部操作。Type ごとの config は作らない |
| `θ(p)` | `p` から決まる内部係数（固定の滑らかな写像） |

## 2. 候補モデル

### 2.1 `sh_inhibitor`（主候補）

Swift–Hohenberg + 遅い局所インヒビター `H`:

```text
u_t = r(p) u - (1 + ∇²/k₀²)² u + g(p) u² - u³ - s(p) H u
H_t = α(p) u₊² - δ(p) H + D_H ∇² H
```

- 固定 `k₀ = 2π / wavelength` が pit 径（特徴幅）を選択
- `g(p)`: spot → stripe/迷路 選択（高 → 低）
- `s(p)`: IV_B 帯でピークし、IV_V 側で緩む（IV_B でユニットを有限に保ち、IV_V では網の連結を許す）
- 異方性項は**なし**: IV_V は等方な迷路／脳回相であり、方向バイアスは平行ストライプ（偽 IV_V）を生むため撤去済み

実装: [`src/simulate_pit_progression.py`](../src/simulate_pit_progression.py)

### 2.2 `sh_memory`（対抗候補）

同じ SH 核に、**伸長履歴場 E** で局所過伸長を抑える:

```text
u_t = ... - m(p) E u
E_t = ε(p) u₊² - ζ(p) E + D_E ∇² E
```

## 3. 内部係数写像（固定）

`coeffs_from_progression(p, model)` が以下を返す（手調整は Type 別ではなく sigmoid の中心のみ）:

| 係数 | 低 p (~I) | 高 p (~IV_V) |
|------|-----------|--------------|
| `quadratic` | ≈ 1.05 | ≈ 0 |
| `growth` | ≈ 0.26 | ≈ 0.13 |
| `inhibitor_strength` | ≈ 0 | IV_B 帯でピーク → 緩む |
| `anisotropy` | 0 | 0（撤去済み） |

## 4. 評価パイプライン

```text
simulate_pit_progression.py  →  場 u の時間積分
        ↓
pit_goal_metrics.py         →  幅・枝長・迷路網（labyrinth）・安定性
        ↓
evaluate_pit_goal.py        →  p スイープ / ramp / 合否 report
```

### 4.1 ゴール指標（`pit_goal_metrics.py`）

| 軸 | 指標 |
|----|------|
| 幅 | `feature_width_median`, `feature_width_p90`, `width_ratio_to_reference` |
| 分岐 | `branch_density`, `branch_arm_count`, `branch_arm_density` |
| IV_B vs IV_V | `labyrinth_score`, `background_island_count`, `longest_component_fraction`, `component_density`（strip） |
| 安定性 | `longest_trend`, `coarsening_detected`, `is_stationary` |

### 4.2 形態ラベル

`classify_morphology()` が I / IIIL / IV_B / IV_V / fail / empty を返す。  
`evaluate_goal_sequence()` が p スイープ全体の順序・幅・branch-before-elongation を判定する。

## 5. 実行方法

```bash
# 短い確認
make pit-goal-smoke

# 独立 p スイープ（同一 IC）
make pit-goal-sweep

# 同一場で p を時間ランプ
make pit-goal-ramp

# 2 候補 + buckling ベースライン比較
make pit-goal-compare

# 空間勾配 p(x) — 1 run で連続遷移（主戦略）
make pit-goal-spatial

# 単体テスト
make test
```

### 5.1 空間勾配モデル（`simulate_pit_spatial.py`）

独立 p スイープは各 `p` で別 attractor に収束し、**相転移的ジャンプ**（点 → 全面 stripe）になる。  
pptx スライド 6 の参照（[`goal_rd_sweep_i_to_iv.png`](images/goal_rd_sweep_i_to_iv.png)）は **1 枚の場で p が左→右に変化**する例である。

| 要素 | 内容 |
|------|------|
| `p(x)` | 水平 cosine feather、左 0 → 右 1 |
| 線形 SH 項 | 一様 FFT（数値安定） |
| 非線形・インヒビター | real-space、係数は `p(x)` から（異方性・stripe バイアスなし） |
| **crypt fission** | 20 step ごとに骨格長 > L(p) のユニットへ主軸垂直のくびれを入れて分裂（`pit_fission.graded_fission_mask`）。`L(p) = 0.66λ·exp(3(p−0.25))`、`p ≥ 0.78` で停止し IV_V 網の連結を許す |
| 合格 | 8 分割 strip に I / IV_B / IV_V が揃い、strip ラベル列が単調、隣接 strip は同相か隣接相のみ（fail strip なし） |

SH の spot→stripe 遷移には**有限長ロッドのアトラクタが存在しない**（spot = 波長 1 個分、stripe = 無限長のみ。中間の短管は伸び続ける過渡状態）。インヒビター H も持続場所を一様に弱らせるだけで長さの物差しを持たない（帯を IIIL 側へ広げる実験では伸長 9.2 → 10.9〜14.8 と悪化）。そこで長さ選択を crypt fission（大腸陰窩の実在機構）として明示的に導入し、**幅は k₀、長さは L(p)** という分担にした。導入後、IIIL 帯の `p90_component_elongation` は 9.2 → 3.2（seed 1/3/7/11/42 で 3.2〜5.8）。

出力: `artifacts/pit_spatial/spatial_overview.png`, `spatial_report.json`

```bash
make pit-goal-spatial
```

1 枚サマリー（**現行モデルの実績**。ゴール規範ではない。規範は [`pit_pattern_goal.md`](pit_pattern_goal.md) §9.1）: [`images/goal_all_results_one_page.png`](images/goal_all_results_one_page.png)

## 6. 現状の診断

### 6.1 独立 p スイープ — ラベル順は通るが視覚的に非連続（撤回）

`sh_inhibitor` の 25 点スイープは `goal_report.json` で `passed: true` となり得るが、**隣接 p 間で画素差 ~80 のジャンプ**があり、contact sheet 上は p≈0.42 付近で点 → 全面 stripe に切り替わる。  
**数値合格 ≠ ゴール達成** とし、主評価は空間勾配 1 run に移した。

### 6.2 空間勾配（seed=7, size=200, steps=5500）

`make pit-goal-spatial` → `spatial_report.json` **`passed: true`**（2026-08 時点、異方性撤去 + crypt fission 導入後）:

- **左**: 規則的 round spot（I）
- **p ≈ 0.4**: 短管（IIIL）。fission により伸長は幅の 3〜5 倍で頭打ち
- **p ≈ 0.55**: 有限の短い分岐ユニット（IV_B、インヒビター帯 + fission）
- **右**: 密な等方迷路／脳回網（IV_V、fission 停止域）。平行ストライプは `fail` 判定
- seed 1/3/7/11/42 すべて `passed: true`、相系列 I → IIIL → IV_B → IV_V は単調

参照: 上記 `goal_all_results_one_page.png`、pptx スイープ [`goal_rd_sweep_i_to_iv.png`](images/goal_rd_sweep_i_to_iv.png)。  
IV_V の規範像は pptx スライド 6 中央の幅一定 labyrinth（[`pit_pattern_goal.md`](pit_pattern_goal.md)）。

### 6.3 旧スイープのラベル系列（参考のみ）

`sh_inhibitor` 11 点スイープではラベル順 I → IIIL → IV_B → IV_V が出る p 帯がある（視覚的連続性は別問題）:

- **I**（p ≈ 0–0.2）: 規則的 spot、幅 ≈ 7 px
- **IIIL**（p ≈ 0.3）: 短管化、分岐なし
- **IV_B**（p ≈ 0.4）: `branch_arm_density` 上昇
- **IV_V**（p ≳ 0.5）: 連結度（`longest_component_fraction` / `labyrinth_score`）上昇

参照画像: [`images/goal_sweep_sim_i_iiil_ivb_ivv.png`](images/goal_sweep_sim_i_iiil_ivb_ivv.png)

## 7. 既存モデルとの位置づけ

| モデル | 役割 |
|--------|------|
| `simulate_pit_progression.py` | **主系列**（本ゴール） |
| `evaluate_pit_steps.py` | Step 比較・式変更実験（参考） |
| `simulate_pit_buckling.py` | 非 RD ベースライン（`evaluate_pit_goal.py compare`） |

## 8. 次の改善候補

1. IIIL 遷移帯の幅を広げる（`elongation(p)` のピーク位置・振幅）
2. `θ(p)` / `θ(p(x))` の自動探索
3. seed 頑健性（`make pit-goal-robustness` を spatial に拡張）
4. 分岐検出の skeleton 閾値校正

関連: [`pit_pattern_goal.md`](pit_pattern_goal.md), [`pit_model_evaluation.md`](pit_model_evaluation.md)
