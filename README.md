# nanoPoet

> 在一台 MacBook 上体验 GPT 风格 LLM 的完整训练流程

**nanoPoet** 是一个学习性质的中国古典诗词生成项目，让你在个人电脑上就能完整体验从数据准备、分词器训练、预训练、中间训练、监督微调到强化学习的全流程。

## 🎯 项目目标

训练一个能够根据**用户指定条件**生成古典诗词的语言模型：

- ✅ 支持多种格式：五言绝句、七言绝句、五言律诗、七言律诗、宋词（各词牌）
- ✅ 条件生成：指定作者风格、诗词形式、标题
- ✅ 完整流程：Tokenizer → PreTrain → MidTrain → SFT → RL
- ✅ 低资源要求：可在 MacBook（CPU/MPS）上训练

**注意**：因为使用的数据集和模型规模都非常小，所以没法实现如 ChatGPT 一般的对话体验。

## 📄 License

MIT License - 自由使用和修改

## 🙏 致谢

- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [nanochat](https://github.com/karpathy/nanochat)
- [Andrej Karpathy - GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

**Happy Poetry Training! 🎭📜✨**
