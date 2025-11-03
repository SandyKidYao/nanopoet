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

def filter_by_author(p: dict):
    if p["author"] in AUTHOR_S + AUTHOR_T:
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
        f"{AUTHOR_START}{poem['author']}{AUTHOR_END}" if poem.get('author') else "",
        f"{STYLE_START}{poem['style']}{STYLE_END}" if poem.get('style') else "",
        f"{TITLE_START}{poem['title']}{TITLE_END}" if poem.get('title') else "",
        f"{CONTENT_START}{poem['content']}{CONTENT_END}"
    ])


def encode_poem_prompt(author: str = None, style: str = None, title: str = None):
    return "".join([
        f"{BEGIN}",
        f"{AUTHOR_START}{author}{AUTHOR_END}" if author else "",
        f"{STYLE_START}{style}{STYLE_END}" if style else "",
        f"{TITLE_START}{title}{TITLE_END}" if title else "",
    ])


def decode_poem_str(encoded_str: str) -> dict:
    """
    从 encode_poem 生成的字符串中解码出原始的 dict 数据
    """
    poem = {}

    # 移除开头的 BEGIN 标记
    if encoded_str.startswith(BEGIN):
        encoded_str = encoded_str[len(BEGIN):]

    # 定义字段映射：(起始标记, 结束标记, 字段名)
    fields = [
        (AUTHOR_START, AUTHOR_END, 'author'),
        (STYLE_START, STYLE_END, 'style'),
        (TITLE_START, TITLE_END, 'title'),
        (CONTENT_START, CONTENT_END, 'content'),
    ]

    # 逐个提取字段
    for start_token, end_token, field_name in fields:
        start_idx = encoded_str.find(start_token)
        if start_idx != -1:
            # 找到起始标记，查找对应的结束标记
            end_idx = encoded_str.find(end_token, start_idx + len(start_token))
            if end_idx != -1:
                # 提取字段内容
                content = encoded_str[start_idx + len(start_token):end_idx]
                poem[field_name] = content
                # 移除已处理的部分（可选，用于优化后续查找）
                # encoded_str = encoded_str[:start_idx] + encoded_str[end_idx + len(end_token):]

    return poem

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
