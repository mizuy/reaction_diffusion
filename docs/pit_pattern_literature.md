# pit pattern（工藤分類）文献レビュー

このドキュメントは、本リポジトリの反応拡散モデル（`src/pit.py` および評価スクリプト `src/evaluate_pit_steps.py`）が再現対象としている「pit pattern」について文献を調査し、その定義・形態・臨床的意義を整理したものである。モデル評価の考え方は `docs/pit_model_evaluation.md` にあり、本ドキュメントはその医学的背景を補完する。

## 1. pit pattern とは何か

「pit pattern（腺口模様、ピットパターン）」とは、大腸粘膜の表面に開口する腺管（crypt）の開口部（pit）が作る微細な模様のことである。拡大内視鏡（magnifying endoscopy）に色素散布（indigo carmine による chromoendoscopy）や染色（crystal violet / cresyl violet などの staining）を組み合わせて観察する。

この模様の型を分類することで、生検・切除前に病変の組織型（腫瘍か非腫瘍か）や浸潤の深さをその場で推定できる。これが工藤進英（Shin-ei Kudo）らによって体系化された **Kudo pit pattern classification（工藤分類）** である。

- 一次文献: Kudo S, Tamura S, Nakajima T, Yamano H, Kusaka H, Watanabe H. "Diagnosis of colorectal tumorous lesions by magnifying endoscopy." *Gastrointestinal Endoscopy* 1996;44(1):8–14. doi:10.1016/S0016-5107(96)70222-5
- この 1996 年の論文が臨床的分類として最も広く引用される seminal work であり、2050 病変を拡大内視鏡・実体顕微鏡・病理で対応づけている。

## 2. 各 Type の形態と臨床的意義

工藤分類は大きく 5 型（サブタイプを含めて 7 区分）に分かれる。腫瘍・非腫瘍の境界は Type II と Type III の間にある。

| Type | pit の形態 | 主な組織像 | 腫瘍性 |
| --- | --- | --- | --- |
| I | 円形（round）の正常な pit | 正常粘膜 | 非腫瘍 |
| II | 星芒状（stellar）または乳頭状（papillary） | 過形成性ポリープ、鋸歯状病変 | 非腫瘍 |
| IIIS | 正常より小さい管状・類円形 pit | 腺腫、粘膜内癌 | 腫瘍 |
| IIIL | 正常より大きい管状・類円形 pit | 腺腫 | 腫瘍 |
| IV | 樹枝状（branch-like）・脳回状（gyrus-like） | 腺腫、粘膜内癌 | 腫瘍 |
| VI（Vi） | I〜IV の pit が大きさ・形・配列とも不整（irregular） | 粘膜内癌、表層 SM 浸潤癌 | 腫瘍（浸潤性） |
| VN（Vn） | 構造の消失した無構造・無定形（non-structural / amorphous） | 深部 SM 浸潤癌 | 腫瘍（深部浸潤性） |

補足事項:

- Type V は irregular な **Vi** と non-structural な **Vn** に細分される。Vn は深部粘膜下層（SM massive）浸潤癌を強く示唆し、内視鏡切除ではなく外科切除の適応判断に用いられる。
- 浸潤深度診断の観点では、「invasive / non-invasive pattern」という枠組みも用いられる（Matsuda et al., *Am J Gastroenterol* 2008;103:2700–6）。
- 鋸歯状病変（SSA/SSL）に特異性が高い **Type II-O**（open-shape II）という修飾型も提案されている（Kimura et al., *Am J Gastroenterol* 2012;107:460–9）。
- 診断精度についてはメタアナリシス（Li M, et al. *World J Gastroenterol* 2014;20(35):12649. doi:10.3748/wjg.v20.i35.12649）があり、非腫瘍 vs 腫瘍の鑑別で有用とされる一方、微小・小ポリープでは信頼性に限界があるとの報告もある（*Gastrointestinal Disorders* 2024;6(3):44. doi:10.3390/gidisord6030044）。

## 3. 形態学的に見た「模様」の階層

反応拡散モデルの観点では、各 Type を「pit の幾何形」と「pit の配列・秩序」の 2 軸で捉えると対応づけやすい。

- 秩序ある spot 状: Type I（規則的な round pit）
- 秩序ある star / small spot: Type II
- 小〜大の spot / short tubular: Type IIIS, IIIL
- stripe / branching / labyrinth: Type IV（樹枝状・脳回状）
- regime の混在と不整: Type VI（複数の pattern が不整に共存）
- 構造の崩壊: Type VN（模様そのものが消失）

Type I → III → IV → VI の流れは、「規則的な spot」から「stripe / 分岐」を経て「不整で複数 regime の混在」へと、パターンの秩序が段階的に崩れていく過程として読める。これは本リポジトリのステップ評価（`docs/pit_model_evaluation.md`）が Type IV の branching と Type VI の不整を主対象にしている理由と整合する。

## 4. 生物学的背景と反応拡散モデルの妥当性

pit pattern を反応拡散で扱うことには、単なる図形マッチング以上の生物学的動機がある。

- 大腸の crypt（腺管）は、Wnt シグナル（短距離活性化）と BMP シグナル（長距離抑制）による **局所活性化・側方抑制（LALI: local autoactivation–lateral inhibition）** の枠組みで形成・維持されると考えられている。これは Turing 型反応拡散と形式的に等価な機構である。
  - Zhang L, et al. "A reaction–diffusion mechanism influences cell lineage progression as a basis for formation, regeneration, and stability of intestinal crypts." *BMC Systems Biology* 2012;6:93. doi:10.1186/1752-0509-6-93
  - この研究では、BMP の喪失や Wnt の過剰が crypt の分裂（fission）・増殖（multiplication）や fingering pattern を生むことが示されており、pit pattern の異常化と対応づけられる。
- LALI / Turing 機構の理論的基盤:
  - Turing AM. "The Chemical Basis of Morphogenesis." *Phil. Trans. R. Soc. B* 1952. spot / stripe / labyrinth などのパターンが均一状態から自発的に生じる。
  - Gierer–Meinhardt の activator–inhibitor モデルが LALI の代表例。
- crypt fission と aberrant pit の連続体モデルも提案されている（individual-based / convection-diffusion モデル）。増殖細胞の増加が単一 crypt の二分岐（bifurcation）を引き起こす様子が再現されている。

つまり、pit pattern の morphology（spot・tubular・branching・不整）は、パラメータ（feed / kill、活性化・抑制のバランス、環境不均一性）の変化に応じて Turing 型系が示すパターン遷移として自然に解釈できる。本リポジトリが Gray-Scott モデルの式変更で Type I / III / IV / VI を近似的に再現しようとしているのは、この生物学的知見に沿った妥当なアプローチである。

なお、反応拡散以外の機構（力学的座屈、細胞ベースモデルなど）でも pit pattern は生成できる。反応拡散を用いないモデルの整理とプロトタイプ実装は `docs/non_reaction_diffusion_models.md` を参照。

## 5. 本リポジトリのモデルとの対応

`src/evaluate_pit_steps.py` の各 config は、上記の理解を踏まえて次のように pit pattern に対応づけられている。

- `reference_pit_i`（feed=0.055, k=0.062）: 規則的な spot → Type I 相当
- `reference_pit_iii`（feed=0.045, k=0.066）: 小〜大の spot / short tubular → Type III 相当
- `step0_pit_iv_baseline`（feed=0.033, k=0.056）: labyrinth / stripe → Type IV 候補（ただし線が伸びすぎる課題）
- `step1`〜`step1b`（反応項の飽和 + cubic damping）: 長い stripe を切って短い branching を増やす → Type IV の改善
- `step2`（静的環境場 C）: 空間的不均一性で線を短く複雑に
- `step3_pit_vi_dynamic_environment`（B と結合した動的環境場 C）: 複数 regime の共存で Type VI の不整を表現

評価指標（`branch_density`, `long_line_score`, `vi_irregularity_score` など）は、上記の「Type IV の branching」と「Type VI の不整」という医学的特徴を定量化する試みである。

## 6. 注意点・限界

- 工藤分類は本来、拡大内視鏡による生体観察のための臨床分類であり、反応拡散モデルはその形態的特徴（spot / stripe / branching / 不整 / 無構造）を近似するに過ぎない。モデル出力は医学的な Type 判定そのものではない。
- 特に Type II（星芒状）と Type VN（無構造）は、単純な activator–inhibitor では表現しづらく、本リポジトリでも現時点では主評価対象から外している。
- 分類の観察者間一致率や微小病変での診断精度には限界があることが報告されており、pit pattern は他の所見（vascular pattern, JNET/NICE 分類など）と併用されるのが実臨床である。

## 7. 主要参考文献

1. Kudo S, et al. Diagnosis of colorectal tumorous lesions by magnifying endoscopy. *Gastrointest Endosc* 1996;44:8–14. (一次文献・工藤分類)
2. Matsuda T, et al. Efficacy of the invasive/non-invasive pattern by magnifying chromoendoscopy to estimate the depth of invasion. *Am J Gastroenterol* 2008;103:2700–6.
3. Kimura T, et al. A novel pit pattern identifies the precursor of colorectal cancer derived from sessile serrated adenoma. *Am J Gastroenterol* 2012;107:460–9. (Type II-O)
4. Li M, et al. Kudo's pit pattern classification for colorectal neoplasms: A meta-analysis. *World J Gastroenterol* 2014;20(35):12649.
5. Zhang L, et al. A reaction–diffusion mechanism influences cell lineage progression as a basis for formation, regeneration, and stability of intestinal crypts. *BMC Syst Biol* 2012;6:93.
6. Turing AM. The Chemical Basis of Morphogenesis. *Phil Trans R Soc B* 1952;237:37–72.
