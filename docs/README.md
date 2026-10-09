# docs の一覧

## 最初に読むもの

| 文書 | 内容 |
|---|---|
| [`pit_pattern_goal.md`](pit_pattern_goal.md) | **ゴール定義**（答案見本・判定基準）。新しいモデルは必ずこれで判定する |
| [`pit_model_evaluation.md`](pit_model_evaluation.md) | Kudo pit pattern を反応拡散モデルで評価する考え方・指標 |
| [`pit_pattern_literature.md`](pit_pattern_literature.md) | pit pattern（工藤分類）の文献レビュー |
| [`serrated_pitpattern.pptx`](serrated_pitpattern.pptx) | 「“鋸歯状”Pit patternを整理する」スライド（形態の参照元） |

## models/ — モデルごとの結果・設計メモ

| 文書 | 内容 |
|---|---|
| [`models/pit_groove_model.md`](models/pit_groove_model.md) | **畝–溝モデル**（2026-10-10、現在の到達点）。捨てた候補の一覧と比較図も含む |
| [`models/pit_microphase_results.md`](models/pit_microphase_results.md) | 非局所相分離（Ohta–Kawasaki 型）モデルの四形態系列 |
| [`models/pit_microphase_high_m_results.md`](models/pit_microphase_high_m_results.md) | 同モデルで m を増やしたときの IV_V と反転 spot |
| [`models/pit_activator_inhibitor_results.md`](models/pit_activator_inhibitor_results.md) | 二場の活性化・抑制モデルの初回結果 |
| [`models/pit_ring_results.md`](models/pit_ring_results.md) | 管腔–上皮輪郭モデルの初回結果 |
| [`models/pit_progression_model.md`](models/pit_progression_model.md) | Swift–Hohenberg 系の連続遷移モデル（空間勾配版を含む）の設計メモ |
| [`models/non_reaction_diffusion_models.md`](models/non_reaction_diffusion_models.md) | 反応拡散以外（座屈・Voronoi など）のモデル |
| [`models/cell_mechanics_models.md`](models/cell_mechanics_models.md) | 細胞力学モデル（CPM・off-lattice 粒子）の設計メモ |

## protocols/ — 検証の手順

| 文書 | 内容 |
|---|---|
| [`protocols/pit_independent_initialization_protocol.md`](protocols/pit_independent_initialization_protocol.md) | **独立初期化による検証**（現行の検証の正本） |
| [`protocols/pit_microphase_protocol.md`](protocols/pit_microphase_protocol.md) | 非局所相分離モデルの検証計画 |
| [`protocols/pit_ring_protocol.md`](protocols/pit_ring_protocol.md) | 管腔–上皮輪郭モデルの検証計画 |

## concepts/ — 構想・設計案

| 文書 | 内容 |
|---|---|
| [`concepts/pit_mechanochemical_transition_concept.md`](concepts/pit_mechanochemical_transition_concept.md) | 上皮–間質–管腔の力学による遷移の構想 |
| [`concepts/pit_type_iv_bistable_width_concept.md`](concepts/pit_type_iv_bistable_width_concept.md) | Type IV の幅制御（bistable plateau） |
| [`concepts/pit_type_iv_local_inhibitor_concept.md`](concepts/pit_type_iv_local_inhibitor_concept.md) | 局所インヒビター場による長さスケール制御 |
| [`concepts/pit_type_iv_saturation_environment_concept.md`](concepts/pit_type_iv_saturation_environment_concept.md) | 反応飽和・環境場による Type IV / VI |

## images/

答案見本・参照シート・反例（規範）と、各モデルの結果図（実績）が同じフォルダにある。どれが規範かは [`pit_pattern_goal.md`](pit_pattern_goal.md) §4 を参照。
