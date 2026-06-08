# 経験キャッシュ・ハーネス — 拡張(IEEE向け)実測 findings (2026-06-07)

> PoC(単一タスク・単一モデル, `cache-harness-findings.md`)を、複数タスク×複数モデル×fine-tuning比較へ拡張。
> タスク非依存に汎用化（`cache_harness/task.py` 他）。論文ドラフト=`paper.md`、図=`figs/`、結果=`*_results.json`。

## セットアップ
- タスク: 自作JP感情(3, controlled/LOTO) + SST-2(2,英) + AG News(4,英) + TREC(6,英)。公開ベンチで著者バイアス回避。
- モデル: Qwen2.5 {0.5,1.5,3}B + Qwen3.5-0.8B + Llama3.2 {1,3}B + Gemma2-2B。教師=Qwen2.5-coder。埋め込み=nomic。
- 比較: zeroshot / dyn-char / dyn-embed / rules-only / cache(emb+rules) / kNN(LLM不使用) / hybrid / embed+LR / **LoRA**。
- 統計: Wilson CI, 対応ありMcNemar, macro-F1, 新規層別, temp=0決定論, HPはdevのみ。

## 主要結果（正直版）

### 1. 小型LLMは多クラスで脆弱（動機の確認）
zeroshot: 0.8B → SST-2 88% / JP 91% だが AG News 32.5% / TREC 22.5%。0.5B → TREC **3.3%**。

### 2. キャッシュは「弱いLLM」を有意に底上げ（マルチタスク, 0.8B）
| タスク | zero | dyn-char | dyn-embed | rules | cache | kNN(LLM無) |
|---|---|---|---|---|---|---|
| JP(3) | 91.2 | 91.2 | **100.0** | 85.3 | 79.4 | 64.7 |
| SST-2(2) | 88.3 | 83.3 | 87.5 | 90.0 | 89.2 | 77.5 |
| AGNews(4) | 32.5 | 26.7 | 30.0 | 40.8 | 44.2† | 75.8 |
| TREC(6) | 22.5 | 39.2† | 32.5† | 40.0† | 39.2† | 66.7 |

† McNemar p<0.05（TREC は p<10⁻³）。**文字n-gramは時に悪化、埋め込みは有効。自動ルールは弱モデルに有効・強モデルに有害。**

### 3. 最重要: キャッシュを「非パラメトリック予測器(LLM不使用)」にすると最強
kNN が LLM全アームを圧倒: AG News **75.8%**(vs 44.2), TREC **66.7%**(vs 40.0)。小型では LLM を介さない方が良い。

### 4. スケーリング則: 小さいほど効く
最良キャッシュ−zeroshot gain: JP 0.5B **+38.2** / 1.2B +29.4 / 1.5B +0.0 / 3.1B −2.9。SST-2 0.5B +20.8 / 3B +4.2。

### 5. 再学習との比較（同一0.5Bベース, 同一ラベル）
| タスク | zero | cache-LLM | kNN | embed+LR | LoRA |
|---|---|---|---|---|---|
| JP | 26.5 | 38.2 | 64.7 | 73.5 | **88.2** |
| SST-2 | 66.7 | 69.2 | 77.5 | **92.5** | 90.0 |
| AGNews | 40.0 | 35.8 | 75.8 | **78.3** | 75.8 |
| TREC | 3.3 | 13.3 | 66.7 | **75.8** | 77.5 |

→ **0.5Bでは「LLMを呼ぶ」のが最悪。埋め込み分類器(kNN/logreg)は学習ゼロ/微小で LoRA に匹敵**（SST-2/AGNewsで上回る）、即時更新可。LoRAはJPで優位。

### 6. 記憶でなく汎化（controlled, 初期設定）
LOTO(未知テーマ)で 72→94%, p<0.004, neutral再現40→100%。nn-copy/反転/新規層別/oracle≈distilled が裏付け。
公開ベンチは標準 held-out で汎化担保。

### 7. オンライン生涯学習・非単調則（初期設定）
ゼロ知識→自己蒸留で 92→100%、短絡13→88%、LLM/問 0.87→0.12(7倍安)、遅延6倍速。学習曲線は **K≈8-32 で最適・大プールで悪化**。

## 進化した命題（論文の主張）
小型モデルには、検証済み経験キャッシュ＝**学習不要の統一適応層**。base の力量で2経路を信頼度ルーティング:
(i) 有能なら **LLM＋埋め込み exemplar**（JP-0.8Bで100%）、(ii) 弱いなら **非パラメトリック予測器**（多クラスでLLMを圧倒し LoRA に匹敵）。
zeroshot を候補に含む dev 選択ルータは下振れせず、短絡で再来は0コスト。**効果はモデルが小さいほど大きい**。

## 追加実験（できることは全部やった: banking77 / 統一LOTO / 1.5B LoRA / BibTeX）

### 8. クラス数 vs 精度（決定打, 2→77クラス, `classcount_results.json`, `figs/classcount.png`）
| タスク | クラス | zs-0.5B | zs-0.8B | kNN | embed+LR |
|---|---|---|---|---|---|
| SST-2 | 2 | 66.7 | 88.0 | 75.3 | 88.0 |
| JP | 3 | 26.5 | **91.2** | 67.6 | 79.4 |
| AG News | 4 | 37.2 | 37.2 | 75.7 | **81.1** |
| TREC | 6 | 2.7 | 22.7 | 71.3 | **80.7** |
| banking77 | 77 | 23.5 | 36.5 | 74.5 | **82.0** |

→ **非パラメトリック・キャッシュ(embed+LR)はクラス数にほぼ不変(～80%)。小型LLMはクラス数で崩壊**（77クラスで差 +45～60pt）。例外=JP3クラスは0.8B LLMが最良(91)＝「有能なモデル×少クラス」では LLM 経路。

### 9. 統一パイプライン controlled（同テーマ & **LOTO**, `controlled_results.json`）
LOTO(未知テーマ, n=50): zeroshot 90.0 / dyn_embed 92.0 / cache 92.0 / oracle 94.0 / **corrupt 76.0** / **kNN 56.0** / embed-LR 64.0。
- **重要な OOD ニュアンス**: 未知テーマでは **LLM+exemplar が汎化(92)・kNN は急落(56)**＝§8の「kNN支配(in-dist)」と鏡像。両経路は相補的→信頼度ルーティングが正解。
- コントロール: corrupt 76≪92（ラベルを反映＝推論している）、oracle94≈distilled92（ルーブリック漏洩なし）、novel==全体（記憶でない）。
- クリーンな label-only プロンプトで JP zeroshot が 90-91% に上昇したため JP上の cache の上積みは小（大きな利得は難タスク/小モデル側 §1-5,8）。

### 10. cache vs LoRA を 1.5B でも（`finetune15_results.json`, `figs/finetune_scale.png`）
1.5B: embed+LR が LoRA に匹敵/上回る（SST-2 92.5=92.5, AGNews 78.3>75.0, TREC 75.8>74.2; JP は LoRA/LLM=100）。**学習ゼロで LoRA 相当**がサイズを跨いで成立。1.5BではLLM経路も有能(JP 100)。

### 11. 体裁
`references.bib`（16件: ExpeL/Reflexion/MemPrompt/CBR/Memento/KATE/GPTCache/calibrate/order/distract/RALM/LoRA/SST/AGNews/TREC/banking77）。図9枚 `figs/`。論文 `paper.md`(全章, 結果反映済)。

### 12. 外部・実世界(Kaggle系)データ再検証（`external_results.json`, `figs/external_classcount.png`）
著者バイアスの無い実データ6種で主要知見を外部検証（airlineは **Kaggle CLI で直接取得**、他はHFミラーの実データ）。
| データ | クラス | zs-0.5B | zs-0.8B | LLM+例 | kNN | embed+LR |
|---|---|---|---|---|---|---|
| IMDB | 2 | 52.0 | 50.0 | 56.0 | 80.7 | **81.3** |
| Airline(Kaggle直) | 3 | 44.0 | 61.3 | 56.7 | 48.7 | **64.7** |
| Tweet | 3 | 45.3 | 55.3 | 44.0 | 48.7 | 50.7 |
| Yelp | 5 | 27.3 | 26.7 | 24.0 | 49.3 | **56.7** |
| Emotion | 6 | 22.7 | 33.3 | 34.0 | 44.0 | **54.7** |
| DBpedia | 14 | 21.4 | 72.1 | 64.3 | 90.7 | **94.3** |

→ **embed+LR が6データ中5で最良/同等**。DBpedia-14で **94.3%**（素LLM 21-72%）。**小型LLMは実データでも弱い**（IMDB二値で~50%=偶然）。
唯一 **短く雑なツイート(3クラス generic)は全手法~50%**（埋め込みも分離難）= 正直な弱点。Kaggle airline(より話題的)は embed+LR 64.7 で勝つ。
**H1(キャッシュ頑健)・H2(小型LLM脆弱)が外部再現。** 注: 非パラメトリック経路の強さは埋め込みモデル(nomic)由来の寄与を含む（透明に明記）。Kaggle認証は `~/.kaggle/kaggle.json`(email+KGATキー)で疎通。

### 13. 複数埋め込みモデル比較（`embedders_results.json`, `figs/embedders.png`）— 「強さは埋め込み由来」caveatの定量化
embed-LR 精度を 4 embedder × 7データで比較（平均）: nomic 69.3 / bge-m3 69.4 / mxbai 70.8 / **all-minilm(45MB) 60.7**。
→ **実用embedder 3種は ~2pt 以内で一致**＝キャッシュの優位は nomic 固有でなく頑健。**小型embedderは~9pt低下**（多クラスで特に, Yelp/emotion 15-20pt）＝embedder品質は効く。
ただし **弱いall-minilmでも多クラスでは素の小型LLMを上回る**（DBpedia 88.6 / banking77 85.3 vs LLM 21-72）。質的主張「多クラス×sub-1Bはキャッシュ予測」はembedder選択に頑健。

### 14. 提出可能化（LaTeX/PDF）
`build_pandoc_md.py`(\cite→[@], \ref無害化, 画像幅付与) → pandoc+xelatex(STIX Two Text) で **`paper.pdf`(18頁)** と **`paper.tex`(91KB)** を生成。`references.bib`(別名込みで全29引用解決)。

## 限界
単一著者JP・小n、dev小で選択にノイズ、分類タスク限定。非パラメ強さは**embedder由来の寄与を含む（§13で定量化: 実用embedderは頑健・弱embedderは低下）**。短く雑な実ツイートでは優位が縮む。単一embedderでない比較は実施済(4種)。JP公開データは datasets 4.x script 非対応で自作JP代替。生成/多段タスクは今後。
