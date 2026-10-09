# reaction_diffusion — 大腸 pit pattern の形態シミュレーション

拡大内視鏡で見る大腸 pit pattern（工藤分類）の進行 **Type I → IIIL → IV_B → IV_V** を、1 本の連続パラメータで再現するモデルを探すプロジェクト。

- **ゴール（まず読む）**: [`docs/pit_pattern_goal.md`](docs/pit_pattern_goal.md)。答案見本は [`docs/images/pit_ideal_sweep.png`](docs/images/pit_ideal_sweep.png)
- **現在の到達点**: 畝–溝モデル [`docs/models/pit_groove_model.md`](docs/models/pit_groove_model.md)
- **文書の一覧**: [`docs/README.md`](docs/README.md)

## セットアップ

```bash
make setup          # = uv sync --extra dev（src/pitpattern を編集可能な形でインストール）
make test           # pytest
```

Python 3.12 / uv。依存は `pyproject.toml`（numpy, opencv-python, pillow, pygame, pygame-gui）。

## 構成

```text
src/pitpattern/
  metrics/      形態指標（幅・分岐・連結・三叉の数え方）
  models/       各候補モデルのシミュレーション（simulate_pit_*.py）
  evaluation/   seed 統計・検証・集計（evaluate_* / validate_* / summarize_*）
  figures/      図の生成（docs/images に出力するもの）
tests/          pytest
docs/           ゴール・評価法・文献、models/・protocols/・concepts/、images/
legacy/         2019〜2024 の pygame 対話ツール（Gray–Scott の手動探索）
artifacts/      計算結果（git 管理外。再計算で作り直せる）
```

## よく使うコマンド

```bash
# 現行モデル（畝–溝モデル）
uv run python -m pitpattern.models.simulate_pit_groove sweep --output artifacts/pit_groove/try --seeds 1 --drives -0.3 0 0.15
make pit-groove-explore        # 探索 seed 1/7/42
make pit-groove-validate       # 検証 seed 100–119 + 集計 + 図

# 図
make target-figures            # 答案見本・参照シート・シェーマ
make model-comparison          # 候補モデル 9 種の比較図

# 旧ツール（pygame のウィンドウが開く）
make run                       # legacy/pit.py
```

スクリプトは `python -m pitpattern.<サブパッケージ>.<モジュール>` でも、`python src/pitpattern/.../xxx.py` でも実行できる（`make setup` 後）。

## 新しいモデルを足すとき

1. `src/pitpattern/models/simulate_pit_<名前>.py` に実装し、指標は `pitpattern.metrics` を使う。
2. 結果は `artifacts/<名前>/` に出す（git に入れない）。
3. ゴール文書 §3 の基準で判定し、結果と限界を `docs/models/` に書く。検証は [`docs/protocols/pit_independent_initialization_protocol.md`](docs/protocols/pit_independent_initialization_protocol.md) に従う。
4. 規範の画像（答案見本など）と、シミュレーション結果の画像を混同しない。
