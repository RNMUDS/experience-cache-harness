"""日本語感情分類データセット — テーマ構造つき（汎化を正しく測るための設計）。

設計意図（レビュー反映）:
  - 各「テーマ」は 1 ラベルに属し、表現の異なる複数バリアントを持つ。
  - 同テーマ分割: 同じテーマを train/dev/test に散らす（既知テーマ・未知表現の汎化）。
  - LOTO 分割: テーマ丸ごとを test 専用にする（未知テーマへの汎化＝記憶では解けない）。
  - neutral は弱点クラスなので厚め。さらに in_rule（営業時間/価格/寸法…の定型）と
    out_of_rule（地理/言語/歴史/科学の一般的事実）に分け、ルールの過一般化を検査できる。
  - バリアントは語彙を意図的に変え、表面 n-gram の単純コピーで解けないようにする。
"""
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass

from .config import LABELS


@dataclass(frozen=True)
class LabeledItem:
    text: str
    label: str
    theme: str
    in_rule: bool = True  # neutral の定型カテゴリ内か（out_of_rule 検査用）


@dataclass(frozen=True)
class Theme:
    name: str
    label: str
    in_rule: bool
    texts: tuple[str, ...]


# ----------------------------------------------------------------------------
# POSITIVE（明確な肯定的評価）
# ----------------------------------------------------------------------------
_POSITIVE = [
    Theme("service_courtesy", "positive", True, (
        "対応がとても丁寧で大満足です。",
        "スタッフの心配りが行き届いていて気持ちよく過ごせました。",
        "受付の方の応対が親切で、来てよかったと思いました。",
        "細やかな気遣いに感激しました。",
        "問い合わせへの返答が手厚くて安心できました。",
    )),
    Theme("product_delight", "positive", True, (
        "想像以上に使いやすくて感動しました。",
        "期待していた以上の出来で胸が躍りました。",
        "触った瞬間にこれは良いと確信できる完成度でした。",
        "毎日使うのが楽しみになるほど気に入っています。",
        "買って以来ずっと手放せないお気に入りです。",
    )),
    Theme("food_taste", "positive", True, (
        "料理は絶品で、また来たいです。",
        "一口目から思わず笑顔になる美味しさでした。",
        "味付けが見事で、最後まで飽きずに楽しめました。",
        "素材の旨みが活きていて感動的な味でした。",
        "デザートまで完璧で大満足の食事でした。",
    )),
    Theme("delivery_glad", "positive", True, (
        "配送が早くて助かりました。",
        "注文した翌日に届いて驚くほど迅速でした。",
        "思ったよりずっと早く手元に届いて嬉しかったです。",
        "発送の速さに感心しました、また頼みたいです。",
        "急ぎだったので即日対応してもらえて本当に助かりました。",
    )),
    Theme("design_beauty", "positive", True, (
        "デザインが美しく、買ってよかった。",
        "見た目が洗練されていて部屋に置くだけで気分が上がります。",
        "色合いも形も上品で、ひと目で気に入りました。",
        "細部まで美しく仕上げられていてうっとりします。",
        "飾っておきたくなるほど造形が綺麗です。",
    )),
    Theme("value_cospa", "positive", True, (
        "コスパ最高、文句なしです。",
        "この値段でこの質なら大満足、お買い得でした。",
        "価格以上の価値があり大変お得に感じました。",
        "安いのに作りがしっかりしていて得した気分です。",
        "財布にやさしいのに満足度が高く言うことなしです。",
    )),
    Theme("staff_warm", "positive", True, (
        "スタッフの笑顔が素敵で気持ちよかった。",
        "明るく迎えてくれて来店した瞬間から心地よかったです。",
        "従業員の朗らかな雰囲気に癒やされました。",
        "皆さん感じが良くて温かい気持ちになりました。",
        "終始にこやかに接してくれて居心地が良かったです。",
    )),
    Theme("quality_exceeded", "positive", True, (
        "期待を超える品質でした。",
        "想定よりはるかに丈夫でしっかりした作りに驚きました。",
        "細工が緻密で値段からは考えられない高級感です。",
        "縫製も素材も一級品で大満足の仕上がりでした。",
        "品質が群を抜いていて期待を裏切りませんでした。",
    )),
    Theme("recommend_again", "positive", True, (
        "また絶対に利用したい、人にもすすめたいお店です。",
        "友人にも自信を持って紹介できる素晴らしさでした。",
        "リピート確定、次回もここに決めています。",
        "周りにおすすめしたくなる満足度の高さでした。",
        "迷っている人にはぜひ試してほしいと心から思います。",
    )),
    Theme("comfort_relief", "positive", True, (
        "安心して任せられて、終始快適でした。",
        "不安なく過ごせて、来て本当によかったです。",
        "ゆったりとくつろげて満ち足りた時間でした。",
        "丁寧に説明してもらえて心から安心できました。",
        "落ち着いた空間で気持ちよくリラックスできました。",
    )),
]

# ----------------------------------------------------------------------------
# NEGATIVE（明確な否定的評価）
# ----------------------------------------------------------------------------
_NEGATIVE = [
    Theme("wait_too_long", "negative", True, (
        "待ち時間が長すぎて最悪だった。",
        "一時間以上も待たされてうんざりしました。",
        "順番がなかなか来ず、いらだちが募りました。",
        "案内が遅く、ずっと放置されて不快でした。",
        "予約したのに長々と待たされ、時間の無駄でした。",
    )),
    Theme("broke_quickly", "negative", True, (
        "すぐに壊れてがっかりした。",
        "使い始めて数日で故障し、品質を疑います。",
        "一週間も経たずに動かなくなり呆れました。",
        "届いてすぐ不具合が出て、もう信用できません。",
        "あっという間に壊れて、お金を捨てたようなものです。",
    )),
    Theme("rude_staff", "negative", True, (
        "店員の態度が悪くて不快でした。",
        "応対がぞんざいで気分を害しました。",
        "高圧的な物言いに腹が立ちました。",
        "無愛想で見下したような接客にうんざりしました。",
        "こちらの質問を雑にあしらわれて不愉快でした。",
    )),
    Theme("overpriced", "negative", True, (
        "値段の割に品質が低い。",
        "高いお金を払ったのに中身が伴っていません。",
        "価格に見合わないお粗末な作りでがっかりです。",
        "ぼったくりに近く、コスパが悪すぎます。",
        "この程度の質でこの価格は納得できません。",
    )),
    Theme("never_again", "negative", True, (
        "二度と利用したくない。",
        "もう金輪際ここには来ないと決めました。",
        "他人にすすめる気にはとてもなれません。",
        "リピートはあり得ない、心底失望しました。",
        "金輪際関わりたくないと思うほどの体験でした。",
    )),
    Theme("misleading", "negative", True, (
        "説明と違っていて騙された気分。",
        "宣伝と実物がかけ離れていて裏切られました。",
        "聞いていた話と全く異なり不信感しかありません。",
        "誇大な広告に乗せられた自分が情けないです。",
        "記載内容と中身が違い、欺かれたようで腹立たしい。",
    )),
    Theme("noisy_unusable", "negative", True, (
        "音がうるさくて使い物にならない。",
        "騒音がひどく、とても実用に耐えません。",
        "動作音が大きすぎて夜は使えたものではない。",
        "けたたましい音で集中できず役に立ちません。",
        "雑音が常に鳴っていてストレスでしかないです。",
    )),
    Theme("slow_support", "negative", True, (
        "返品対応が遅くてイライラした。",
        "問い合わせの返事が来ず、放置されて困りました。",
        "サポートがいつまでも進まず辟易しました。",
        "連絡しても音沙汰がなく、対応の遅さに憤りました。",
        "手続きが滞り、何日も待たされて不満です。",
    )),
    Theme("dirty_unhygienic", "negative", True, (
        "店内が不潔で居心地が悪かった。",
        "テーブルがべたついていて衛生面が心配でした。",
        "床にゴミが落ちていて清潔感がまるでなかった。",
        "手入れが行き届かず、汚れが目立って残念でした。",
        "水回りが汚く、とても利用する気になれませんでした。",
    )),
    Theme("disappointed", "negative", True, (
        "期待外れで本当にがっかりしました。",
        "楽しみにしていただけに落胆が大きかったです。",
        "思い描いていたものと程遠く、失望しました。",
        "わざわざ足を運んだのに残念な結果でした。",
        "高い期待を裏切られ、肩を落として帰りました。",
    )),
]

# ----------------------------------------------------------------------------
# NEUTRAL — in_rule（営業時間/価格/寸法… の客観的・定型的事実）
# ----------------------------------------------------------------------------
_NEUTRAL_IN_RULE = [
    Theme("business_hours", "neutral", True, (
        "営業時間は午前9時から午後6時までです。",
        "受付は朝10時に始まり夜8時に閉まります。",
        "当店は平日のみ正午から夕方5時まで開いています。",
        "開館は8時、閉館は20時となっております。",
        "土日は10時から19時の間ご利用いただけます。",
    )),
    Theme("price_statement", "neutral", True, (
        "本体価格は税込で12,000円です。",
        "一個あたり380円で販売しています。",
        "月額料金は980円に設定されています。",
        "送料は全国一律500円かかります。",
        "入場料は大人1,500円、子ども700円です。",
    )),
    Theme("dimensions_spec", "neutral", True, (
        "本体の重さは約1.2キログラムです。",
        "画面の大きさは6.1インチです。",
        "幅30センチ、奥行き20センチの設計です。",
        "容量は500ミリリットルとなっています。",
        "対応電圧は100ボルトから240ボルトです。",
    )),
    Theme("shipping_time", "neutral", True, (
        "商品は3営業日以内に発送されます。",
        "ご注文から到着まで通常一週間ほどです。",
        "在庫があれば翌日に出荷されます。",
        "配送には地域により2日から4日かかります。",
        "発送は毎週月曜と木曜に行っています。",
    )),
    Theme("location_access", "neutral", True, (
        "店舗は駅から徒歩5分の場所にあります。",
        "会場は市役所の隣に位置しています。",
        "最寄りのバス停から歩いて3分です。",
        "駐車場は建物の地下一階にあります。",
        "入口は北側、二番出口を出てすぐです。",
    )),
    Theme("color_variants", "neutral", True, (
        "この製品は3色展開です。",
        "カラーは黒、白、青の3種類があります。",
        "色のバリエーションは全部で五つです。",
        "赤と緑の2色からお選びいただけます。",
        "限定色を含めて六種類をご用意しています。",
    )),
    Theme("file_size", "neutral", True, (
        "アプリのサイズは約50MBです。",
        "ダウンロード容量はおよそ120メガバイトです。",
        "インストールには2ギガバイトの空きが必要です。",
        "データ通信量は1回あたり約8MBです。",
        "ファイルの大きさは300キロバイト程度です。",
    )),
    Theme("support_days", "neutral", True, (
        "サポートは平日のみ対応しています。",
        "問い合わせ窓口は月曜から金曜まで稼働しています。",
        "祝日を除く毎日サポートを受け付けています。",
        "電話対応は火曜と木曜に限られます。",
        "土日祝はサポート業務をお休みしています。",
    )),
    Theme("manual_included", "neutral", True, (
        "パッケージには説明書が同梱されています。",
        "箱の中に保証書と取扱説明が入っています。",
        "付属品として充電ケーブルが一本付きます。",
        "同梱物は本体、説明書、予備の電池です。",
        "化粧箱には専用の収納ポーチが含まれます。",
    )),
    Theme("schedule_event", "neutral", True, (
        "会議は明日の14時に開催されます。",
        "説明会は来週の水曜に予定されています。",
        "次回の点検は月末に実施されます。",
        "イベントは10月3日の正午に始まります。",
        "総会は毎年6月の第二金曜に行われます。",
    )),
]

# ----------------------------------------------------------------------------
# NEUTRAL — out_of_rule（定型カテゴリ外の一般的事実。ルール過一般化の検査用）
# ----------------------------------------------------------------------------
_NEUTRAL_OUT_RULE = [
    Theme("geography_fact", "neutral", False, (
        "富士山の標高は3776メートルです。",
        "信濃川は日本で最も長い川です。",
        "琵琶湖は滋賀県の中央に広がっています。",
        "この島は本州の南方に位置しています。",
        "利根川は関東平野を東へ流れています。",
    )),
    Theme("language_fact", "neutral", False, (
        "この単語はラテン語に由来しています。",
        "ひらがなは漢字を崩して作られた文字です。",
        "英語の主語は通常文の先頭に置かれます。",
        "この方言では語尾が独特に変化します。",
        "アルファベットは26文字で構成されています。",
    )),
    Theme("history_fact", "neutral", False, (
        "江戸時代はおよそ260年続きました。",
        "この城は16世紀に築かれたものです。",
        "鉄道が初めて開通したのは明治の初めです。",
        "その条約は二つの国の間で結ばれました。",
        "この寺は奈良時代に建立されたとされています。",
    )),
    Theme("science_fact", "neutral", False, (
        "水は摂氏100度で沸騰します。",
        "光は真空中を秒速約30万キロで進みます。",
        "植物は日光を使って養分を作ります。",
        "鉄は磁石に引き寄せられる性質があります。",
        "月は地球の周りを約27日で一周します。",
    )),
]

ALL_THEMES: tuple[Theme, ...] = tuple(
    _POSITIVE + _NEGATIVE + _NEUTRAL_IN_RULE + _NEUTRAL_OUT_RULE
)


def all_items() -> list[LabeledItem]:
    items: list[LabeledItem] = []
    for theme in ALL_THEMES:
        for text in theme.texts:
            items.append(LabeledItem(text, theme.label, theme.name, theme.in_rule))
    return items


def _themes_by_label() -> dict[str, list[Theme]]:
    out: dict[str, list[Theme]] = {label: [] for label in LABELS}
    for theme in ALL_THEMES:
        out[theme.label].append(theme)
    return out


@dataclass(frozen=True)
class Split:
    train: tuple[LabeledItem, ...]
    dev: tuple[LabeledItem, ...]
    test: tuple[LabeledItem, ...]
    kind: str  # "same_theme" | "loto"


def split_same_theme(seed: int, n_dev: int = 1, n_test: int = 1) -> Split:
    """各テーマ内でバリアントを train/dev/test に分配（既知テーマ・未知表現の汎化）。"""
    rng = random.Random(seed)
    train: list[LabeledItem] = []
    dev: list[LabeledItem] = []
    test: list[LabeledItem] = []
    for theme in ALL_THEMES:
        idx = list(range(len(theme.texts)))
        rng.shuffle(idx)
        test_idx = set(idx[:n_test])
        dev_idx = set(idx[n_test : n_test + n_dev])
        for i, text in enumerate(theme.texts):
            item = LabeledItem(text, theme.label, theme.name, theme.in_rule)
            if i in test_idx:
                test.append(item)
            elif i in dev_idx:
                dev.append(item)
            else:
                train.append(item)
    return Split(tuple(train), tuple(dev), tuple(test), "same_theme")


def split_loto(seed: int, test_theme_frac: float = 0.3, dev_theme_frac: float = 0.15) -> Split:
    """テーマ丸ごとを dev/test 専用に分離（未知テーマへの汎化＝記憶では不可）。"""
    rng = random.Random(seed)
    train: list[LabeledItem] = []
    dev: list[LabeledItem] = []
    test: list[LabeledItem] = []
    for label, themes in _themes_by_label().items():
        order = list(themes)
        rng.shuffle(order)
        n_test = max(1, round(len(order) * test_theme_frac))
        n_dev = max(1, round(len(order) * dev_theme_frac))
        test_themes = order[:n_test]
        dev_themes = order[n_test : n_test + n_dev]
        train_themes = order[n_test + n_dev :]
        for bucket, themes_in in ((test, test_themes), (dev, dev_themes), (train, train_themes)):
            for theme in themes_in:
                for text in theme.texts:
                    bucket.append(LabeledItem(text, theme.label, theme.name, theme.in_rule))
    return Split(tuple(train), tuple(dev), tuple(test), "loto")


def audit_split(split: Split, sim_fn) -> dict:
    """漏洩監査: train↔test の完全一致 / 編集距離比 / 検索類似度の分布を返す。

    sim_fn(test_text, train_texts) -> 最大類似度(0..1)。検索器のコサインを渡す想定。
    編集距離比は difflib（検索指標とは独立な第二の尺度）。
    """
    train_texts = [it.text for it in split.train]
    train_set = set(train_texts)
    exact = sum(1 for it in split.test if it.text in train_set)

    edit_ratios: list[float] = []
    cos_sims: list[float] = []
    for it in split.test:
        best_edit = max(
            (difflib.SequenceMatcher(None, it.text, t).ratio() for t in train_texts),
            default=0.0,
        )
        edit_ratios.append(best_edit)
        cos_sims.append(sim_fn(it.text, train_texts))

    def _dist(values: list[float]) -> dict:
        if not values:
            return {"max": 0.0, "mean": 0.0, "p90": 0.0}
        ordered = sorted(values)
        return {
            "max": max(values),
            "mean": sum(values) / len(values),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        }

    return {
        "kind": split.kind,
        "n_train": len(split.train),
        "n_dev": len(split.dev),
        "n_test": len(split.test),
        "exact_overlap": exact,
        "edit_ratio": _dist(edit_ratios),
        "retrieval_cosine": _dist(cos_sims),
    }
