# ハーネス・カタログ 〜 小型ローカルモデル (qwen3.5:0.8b) 適性評価

> 作成日: 2026-06-03 / 対象モデル: `qwen3.5:0.8b` (ollama, 1.0GB)
> 目的: ユーザーが用途に応じて Web 上で選択・カスタマイズできる「汎用 AI ハーネス」を、
> 小型ローカルモデル上で構築するための土台として、世の中の優秀なハーネスを棚卸しする。

---

## 0. 背景と中心的な仮説

2026 年のハーネス工学(harness engineering)の合言葉は **「LLM はエージェントシステムの最小の部品」**。
品質を決めるのは、モデルの周りを取り囲む足場 — ループ・ツール定義・観測整形・回復・文脈管理 — である。

小型モデルにとってこれは朗報でもあり、現実でもある:

- **現実(ベンチマーク)**: 信頼できる native tool calling は **27B クラスから**。`sub-4B` は多段タスクで malformed call を多発させる。0.8B は「短い1語を返す分類」なら使えるが、「多段プラン」は素では無理。
- **朗報(実例)**: 強いハーネスがあれば小型モデル(例: Claude 3 Haiku)でも信頼性を出せる。**「小型＋強ハーネス」は「大型＋弱ハーネス」に、信頼性でもコストでも勝つ**。

→ 我々の設計指針: **決定論的な制御プレーンを主役にし、0.8B には「極小・構造的に強制された短い判断」だけをさせる。**

---

## 1. 評価軸（4軸 + 小型モデル適性）

`agent-harness-construction` スキルの4軸を採用し、小型モデル適性スコアを付与する。

| 軸 | 問い | 小型モデルで特に効くか |
|---|---|---|
| **行動空間 (Action Space)** | ツールは少数か / 名前は明示的か / 入力スキーマは狭いか | ◎ ツールが多いほど0.8Bは混乱。1ターン1ツールが理想 |
| **観測 (Observation)** | 出力は `status/summary/next_actions/artifacts` 等で定型化されているか | ◎ 整形された観測が次手を誘導する |
| **回復 (Recovery)** | エラーが「原因＋安全な再試行＋停止条件」を含むか | ◎ エラー文言は小型モデルにとって "プロンプトの一部" |
| **文脈予算 (Context)** | system prompt は最小・不変か / 長文は参照に逃がすか | ◎ 0.8B は長文脈で急速に劣化 |

**小型モデル適性レジェンド:**
`◎` = 小型に必須/最適 ／ `○` = 我々の構築に有用 ／ `△` = 条件付き(0.8Bには重い) ／ `×` = 0.8Bでは非推奨(設計の参考にのみ)

---

## 2. カタログ本体

### A. コーディング Agent（CLI / IDE）— *設計の教科書*

> いずれも「フロンティアモデル前提」。0.8B をそのまま載せると破綻するが、**ACI(エージェント-コンピュータ間インタフェース)設計の手本**として価値が高い。

| ハーネス | 特徴 | 行動空間 | 観測 | 回復 | 文脈 | 0.8B適性 |
|---|---|---|---|---|---|---|
| **Claude Code** (Anthropic) | 5段階の漸進的compaction、subagent隔離、27イベントのhookパイプライン | ◎ 明示的・少数 | ◎ | ◎ | ◎ | × (設計◎) |
| **Aider** (39k★) | ターミナル特化。リポジトリマップ、diff/edit形式が簡潔、自動commit | ○ 編集形式が簡潔 | ○ | ○ | ○ | △ (edit形式は学ぶ価値大) |
| **OpenHands** (旧OpenDevin, 70k★) | Docker隔離サンドボックスで自律実行。plan→execute→observeループ | ○ | ○ | ◎ サンドボックス | △ | × (設計参考) |
| **SWE-agent** (Princeton) | **ACI論文の本家**。「単純さ・効率・簡潔なドキュメント」を原則化 | ◎ 原則そのもの | ◎ | ○ | ○ | × (原則は◎必読) |
| **Cline / Roo Code** | VS Code拡張(5M+導入)。Plan/Actで人間が逐次承認、権限制御 | ○ | ○ | ◎ 承認ゲート | ○ | △ (承認ゲートが補助に) |
| **OpenCode** (~165k★, MIT) | プロバイダ非依存、ローカルモデル対応、洗練TUI。Claude Code代替筆頭 | ○ | ○ | ○ | ○ | △ (ローカル接続の参考) |
| **Goose** (Block) | BYOM、拡張(extension)アーキテクチャ、MCPネイティブ | ○ | ○ | ○ | ○ | △ |
| **Continue / Cursor / Gemini CLI / Qwen Code** | BYOM対応のIDE/CLI群。Qwen Codeは Qwen 系に最適化 | ○ | ○ | ○ | ○ | △ (Qwen Codeは要検証) |

**学ぶべき点**: SWE-agent ACI原則「重要操作を少数アクションに集約」「コマンドは少数オプション＋簡潔なドキュメント」は **0.8B にこそ効く**。承認ゲート(Cline)は per-call 失敗を人間が回収する小型モデルの実用的保険。

### B. Agent フレームワーク / SDK — *制御プレーンの部品*

> 我々が「ユーザーがWebで組み替えるハーネス」を作る際の土台候補。

| フレームワーク | 特徴 | 0.8B適性 | 我々への用途 |
|---|---|---|---|
| **smolagents** (HuggingFace) | `CodeAgent`(コードを書いて実行) と `ToolCallingAgent`。**ollama/transformers/LiteLLMでローカル動作**。MCP/LangChainツール流用可 | ○ | **第一候補**。code-as-actionはJSON tool callより小型に優しい場合あり |
| **LangGraph** | エージェントループを有向グラフ＋型付きstate＋条件分岐＋checkpointで明示モデル化 | ○ | **制御プレーンの骨格**。決定論的フローを組み、LLMは小ノードのみ |
| **Letta (MemGPT)** | 3層メモリ(core/archival/recall)の参照設計。状態を持つ永続エージェント | ○ | **メモリ層の参照実装** |
| **Pydantic AI** | 型付き・構造化出力ファースト。検証エラーの自動フィードバック | ◎ | 出力スキーマ強制に好相性 |
| **OpenAI Agents SDK / Google ADK** | ハンドオフ・ガードレール・評価ハーネス(ADKは117プロンプト評価) | ○ | 評価・ガードレールの設計参考 |
| **AutoGen / CrewAI / TaskWeaver** | マルチエージェント会話 / 役割分担 / planner-executor＋プラグイン | △ | マルチエージェントは0.8Bには重い。役割分割の発想のみ採用 |
| **Composio / Nango** | 250+/700+ APIをエージェント向けアクション化、OAuth管理 | ○ | ツール供給層(必要時) |

### C. 構造化出力 / 制約デコード — *0.8B の生命線* ◎

> **ここが小型モデル成功の最重要レイヤ。** 出力を「お願い」するのではなく、サンプリング段階で**構造的に強制**する。

| 技術 | 仕組み | 0.8B適性 |
|---|---|---|
| **ollama structured outputs** (`format` に JSON Schema) | ollama がデコード時に JSON Schema へ拘束 | ◎ 標準・即採用可 |
| **llama.cpp GBNF grammars** | 文法(BNF)でトークンサンプリングを制約。任意構文を強制 | ◎ ツール呼び出し構文を100%整形 |
| **outlines** | regex/CFG/JSON Schema をデコード層で強制 | ◎ |
| **instructor** | Pydanticモデル → 構造化抽出。検証エラーで自動リトライ | ◎ |
| **XGrammar 等の高速文法エンジン** | 制約デコードの低オーバーヘッド化 | ○ |

**結論**: 0.8B には「JSONを書いてね」ではなく **文法/スキーマ制約で malformed を物理的に不可能にする**。これだけで sub-4B の最大の失敗要因(壊れたcall)が消える。

### D. 小型モデル特化のパターン / 知見 — *実戦テクニック* ◎

実例「小型LLMを信頼させるハーネス」(Claude 3 Haiku)由来の、即適用できる手法:

1. **トークン・エイリアシング (Ref Proxy)** — UUID等を `ref_A3k9Xp2Q` のセッション内別名に変換して見せる。「UUIDを一度も見せなければ、parrotできない」。捏造refは構造化エラーで弾く。
2. **必須ツール契約 (Mandatory Tool Contracts)** — 「終了前に必ず `submit` を呼べ」をループの停止条件として**強制**。`endTurn` をゲートにし、契約違反の終了を拒否。→ 小型モデルの典型失敗「最後のcommitツールを呼ばずに散文で終わる」を封じる。
3. **回復可能な構造化エラー** — throwせず `{error, hint}` を返す。「Unknown ref — search again で新しいrefを取れ」のように**次の一手を文言で指示**。モデルは同ターン内で自動再試行。
4. **行動空間の極小化** — 1ターン1ツール、出力は「短い1語/1スロット」。これは0.8Bでも安定して出せる粒度。
5. **タスク分解** — 多段プランを、各ステップが分類/スロット埋め(=短い出力)で済むように割る。「破綻する多段」を「成功する単発の連鎖」に変換。
6. **code-as-action vs JSON** — smolagents流。状況によりコード生成の方がJSON tool callより小型に安定。要ベンチ。
7. **決定論的制御プレーンが主役** — ループ・分岐・検証・整形はコードで握り、LLMは「次にどれ?」「この値は?」だけ判断。

---

## 3. ベンチマークの現実（正直な数字）

- **複利失敗**: per-call 95% 信頼性でも、8ステップで成功率 ~66%。→ 小型モデルではステップ数を削るほど勝てる。
- **規模の壁**: 信頼できる native tool calling は実質 **27B〜**(例: Llama 3.3 70B ~97%, Qwen3-Coder 30B ~96%, Gemma 4 27B ~95%)。sub-7B は多段で malformed。
- **量子化の床**: `Q4_K_M` が実用下限。tool calling はチャット品質より先に量子化で劣化する。
- **エンタープライズ障害の65%は「ハーネス欠陥」**(Context Drift / Schema Misalignment / State Degradation) — モデルでなく足場が原因。

> **0.8B への含意**: 素の多段自律は狙わない。**「制約デコード＋1ターン1判断＋決定論制御」**で、各LLM呼び出しを「短い分類」に落とし込めば実用域に入る。

---

## 4. 我々のプロジェクト（Web カスタマイズ型ハーネス）への示唆

ユーザーが用途別に Web 上で組み替えられる「AI 制御プレーン」を作る、という目標に対して:

- **アーキテクチャの中心**: `LangGraph` 型の明示グラフ or 自前の状態機械を**制御プレーン**に。ノードの大半は決定論、LLM(0.8B)はごく一部の小ノード。
- **出力の土台**: `ollama structured outputs` / `GBNF` を全 LLM 呼び出しに標準適用(◎必須)。
- **ユーザーがWebで選ぶ単位** = ハーネスの「部品」: ①有効化するツール群(行動空間) ②観測整形テンプレ ③ガードレール/承認ゲート ④メモリ層(Letta型 on/off) ⑤プロンプト/スキル束。
  - これは2026のトレンド「Natural-Language Agent Harness(制御ロジックを移植可能な成果物として外部化)」「middleware hooksでコアを触らず方針を差し込む」と一致。
- **メモリ**: `Letta(MemGPT)` の3層メモリを参照。0.8Bには文脈を持たせず、外部メモリ＋検索で補う。
- **評価**: 用途別テンプレごとに完了率 / retry数 / pass@1 を計測(ADK評価ハーネスの発想)。

**推奨スタータースタック(検証用)**:
`smolagents`(or 自前ループ) × `ollama`(qwen3.5:0.8b) × `structured outputs/GBNF` × 極小ツール3〜5個 × 構造化エラー。
→ まず「単発分類が安定するか」を計測 → 段階的に多段(契約＋endTurnゲート)へ。

---

## 5. 次アクション候補

1. `qwen3.5:0.8b` の **構造化出力(`format` JSON Schema)** の素の安定性を実測(短い分類タスク)。
2. **最小ハーネスのPoC**: 1ツール + 構造化出力 + 構造化エラー + endTurnゲート。
3. smolagents `CodeAgent` vs 自前 `ToolCallingAgent`(JSON) を 0.8B で **A/B**。
4. Web カスタマイズ UI の「部品カタログ」を、本書の §4 の5部品で定義。

---

## 6. 出典 (Sources)

- [Agent Harness Engineering — The Rise of the AI Control Plane (Medium, 2026-04)](https://medium.com/@adnanmasood/agent-harness-engineering-the-rise-of-the-ai-control-plane-938ead884b1d)
- [The Agent Harness: Why the LLM Is the Smallest Part of Your Agent System (MongoDB, 2026-05)](https://www.mongodb.com/company/blog/technical/agent-harness-why-llm-is-smallest-part-of-your-agent-system)
- [Building a harness that makes a small LLM reliable (dev.to, Nikhil Verma)](https://dev.to/nikhilverma/building-a-harness-that-makes-a-small-llm-reliable-5ca9)
- [awesome-harness-engineering (GitHub, ai-boost)](https://github.com/ai-boost/awesome-harness-engineering)
- [Best Local Models for Tool Calling in 2026: Benchmarks (promptquorum)](https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026)
- [SWE-agent: Agent-Computer Interfaces (arXiv 2405.15793)](https://arxiv.org/pdf/2405.15793) / [ACI docs](https://swe-agent.com/1.0/background/aci/)
- [smolagents (GitHub, HuggingFace)](https://github.com/huggingface/smolagents) / [smolagents + Ollama](https://medium.com/@abonia/building-practical-local-ai-agents-with-smolagents-ollama-f92900c51897)
- [Best Open Source AI Coding Agents 2026 (opensourcealternatives.to)](https://www.opensourcealternatives.to/blog/best-open-source-ai-coding-assistants)

> 注: 一部の 2026 年新出ハーネス名(例: Confucius CCA, deepclaude 等)は本エージェントの知識カットオフ(2026-01)以降のため、採用検討時は一次情報での再確認を推奨。
