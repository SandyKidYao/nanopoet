import torch

from nanopoet.common import CharTokenizer
from nanopoet.dataset import load_raw_data, split_data, get_base_dir
from nanopoet.model import GPTLanguageModel
from nanopoet.train_mid import train_mid
from nanopoet.train_pre import train_pre


def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    data = load_raw_data("../raw")
    train, val = split_data(data)
    raw_text = "".join(["".join(list(d.values())) for d in data])
    tokenizer = CharTokenizer(raw_text)
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        emb_size=256,
        block_size=256,
        layer_num=8,
        head_num=8,
        dropout=0.1,
    )

    # pre train
    # batch_size = 64
    # train_pre(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     train_poems=train,
    #     val_poems=val,
    #     batch_size=batch_size,
    #     learning_rate=3e-4,
    #     grad_clip=1.0,
    #     total_epochs=10,
    #     eval_interval=len(train) // batch_size // 2,
    #     eval_iters=len(val) // batch_size,
    #     checkpoint_dir=get_base_dir() + "/checkpoints/pre",
    #     output_path=get_base_dir() + "/pre_train_model.pt",
    # )

    # mid train
    batch_size = 32
    train_mid(
        model=model,
        tokenizer=tokenizer,
        train_poems=train,
        val_poems=val,
        device=device,
        batch_size=batch_size,
        learning_rate=3e-4,
        init_lr_frac=0.5,
        warmdown_start_ratio=0.8,
        final_lr_frac=0.0,
        grad_clip=1.0,
        total_epochs=10,
        eval_interval=len(train) // batch_size // 2,
        eval_iters=len(val) // batch_size,
        checkpoint_dir=get_base_dir() + "/checkpoints/mid",
        output_path=get_base_dir() + "/mid_train_model.pt",
        pretrain_model_path=get_base_dir() + "/pre_train_model.pt",
    )


if __name__ == '__main__':
    main()
