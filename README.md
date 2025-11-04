# nanoPoet

> 在一台 MacBook 上体验 GPT 风格 LLM 的完整训练流程

**nanoPoet** 是一个学习性质的中国古典诗词生成项目，让你在个人电脑上就能完整体验从数据准备、分词器训练、预训练、中间训练、监督微调到强化学习的全流程。
项目中提供了 Notebook 版本的简化代码用于学习，也提供了正式的训练代码以及一个简单体验诗词的生成的小网页应用

## 🎯 项目目标

训练一个能够根据**用户指定条件**生成古典诗词的语言模型：

- ✅ 支持多种格式：五言绝句、七言绝句、五言律诗、七言律诗、宋词（各词牌）
- ✅ 条件生成：指定作者风格、诗词形式
- ✅ 完整流程：Tokenizer → PreTrain → MidTrain → SFT → Reward → RL
- ✅ 低资源要求：可在 MacBook（CPU/MPS）上训练

**注意**：因为使用的数据集和模型规模都非常小，所以没法实现如 ChatGPT 一般的对话体验。只能执行一个特定的文本生成类工作。

## 📁 项目结构

```
nanopoet/
├── raw/              # 原始数据
├── nanopoet/         # 主代码包
│   ├── main.py       # 训练入口
│   ├── model.py      # GPT 模型
│   ├── train_*.py    # 各训练阶段
│   ├── reward.py     # 奖励计算
│   └── app.py        # Web 应用
├── notebook/         # Jupyter 实验笔记
│   ├── 00_prepare.ipynb    # 数据准备
│   ├── 01_model.ipynb      # 模型架构
│   ├── 02_pre_train.ipynb  # 预训练
│   ├── 03_mid_train.ipynb  # 中期训练
│   ├── 04_sft.ipynb        # 监督微调
│   ├── 05_reward.ipynb     # 奖励模型
│   └── 06_rl.ipynb         # 强化学习
└── output/           # 训练产物（模型、checkpoint）
```

## 🚀 快速开始

### 安装依赖
```bash
uv sync
source .venv/bin/activate
```

### 训练模型
编辑 `nanopoet/main.py`，取消注释相应训练阶段：
```python
# 依次取消注释以运行各阶段训练
# pre(...)      # 预训练
# mid(...)      # 中期训练
# sft(...)      # 监督微调
# reward(...)   # 奖励分类器
# rl(...)       # 强化学习
```

运行训练：
```bash
cd nanopoet
python main.py
```

### 启动 Web 应用
取消注释 `main.py` 中的 `start_app(...)`，然后访问 http://localhost:54321

### Jupyter 笔记
```bash
jupyter notebook
# 打开 notebook/ 目录下的笔记探索各阶段
```

## 📄 License

MIT License - 自由使用和修改

## 🙏 致谢

- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [nanochat](https://github.com/karpathy/nanochat)
- [Andrej Karpathy - GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

**Happy Poetry Training! 🎭📜✨**
