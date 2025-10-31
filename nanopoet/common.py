# ========== 基础设置 ===========
STYLE_T = ["七言律诗", "七言绝句", "五言律诗", "五言绝句"]
STYLE_S = ["浣溪沙", "水调歌头", "西江月", "鹧鸪天", "沁园春", "蝶恋花"]
AUTHOR_S = ["苏轼", "辛弃疾", "李清照", "柳永", "欧阳修", "李白", "杜甫", "白居易", "王维", "李商隐", "陆游", "杨万里",
            "黄庭坚", "王安石", "朱熹"]
AUTHOR_T = ["蘇軾", "辛棄疾", "李清照", "柳永", "歐陽修", "李白", "杜甫", "白居易", "王維", "李商隱", "陸游", "楊萬里",
            "黃庭堅", "王安石", "朱熹"]


def filter_poem(p: dict):
    """根据设定的作者和格式，筛选诗词，由于STYLE_T中的诗词数量太大，因此筛选时也做作者限制"""
    if p["author"] in AUTHOR_S + AUTHOR_T:
        return True
    if p["style"] in STYLE_S:
        return True
    return False


def update_poem_author(p: dict):
    """将诗词中的作者从繁体转换成简体"""
    if p["author"] not in AUTHOR_T:
        return p
    new_p = {}
    new_p.update(p)
    new_p["author"] = AUTHOR_S[AUTHOR_T.index(p["author"])]
    return new_p


# ========== 特殊 Token ==========
BEGIN = "B"
PADDING = "P"
UNKNOWN = "U"

AUTHOR_START = "A"
AUTHOR_END = "a"

STYLE_START = "S"
STYLE_END = "s"

TITLE_START = "T"
TITLE_END = "t"

CONTENT_START = "C"
CONTENT_END = "c"

SPECIAL_TOKENS = [
    BEGIN, PADDING, UNKNOWN, AUTHOR_START, AUTHOR_END, STYLE_START, STYLE_END,
    TITLE_START, TITLE_END, CONTENT_START, CONTENT_END
]


def encode_poem(poem):
    return "".join([
        f"{BEGIN}",
        f"{AUTHOR_START}{poem['author']}{AUTHOR_END}" if poem['author'] else "",
        f"{STYLE_START}{poem['style']}{STYLE_END}" if poem['style'] else "",
        f"{TITLE_START}{poem['title']}{TITLE_END}" if poem['title'] else "",
        f"{CONTENT_START}{poem['content']}{CONTENT_END}"
    ])


class CharTokenizer:
    """字符级别的分词器"""

    def __init__(
            self,
            raw_text: str,
            special_tokens: list[str] = SPECIAL_TOKENS
    ):
        self.chars = special_tokens + sorted(list(set(raw_text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [self.stoi.get(d, UNKNOWN) for d in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])
        self.vocab_size = len(self.chars)
