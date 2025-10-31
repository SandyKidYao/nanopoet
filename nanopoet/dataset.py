import json
import os


def get_base_dir():
    if os.environ.get("NANOPOET_BASE_DIR"):
        ndir = os.environ.get("NANOPOET_BASE_DIR")
    else:
        ndir = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(ndir, exist_ok=True)
    return ndir


def load_raw_data(data_dir="./raw"):
    with open(os.path.join(data_dir, "base_poetry_data.jsonl")) as f:
        data = [json.loads(l) for l in f]
    return data


def split_data(data:list[dict]) -> tuple[list, list]:
    """均匀取样：将数据按类型排序后，均分到训练和验证集，比例 9:1"""
    sorted_data = sorted(data, key=lambda x: x["style"])
    train_data = []
    val_data = []
    for i, d in enumerate(sorted_data):
        if i % 10 == 0:  # 每10首中的第1首作为验证集
            val_data.append(d)
        else:  # 其余9首作为训练集
            train_data.append(d)
    return train_data, val_data