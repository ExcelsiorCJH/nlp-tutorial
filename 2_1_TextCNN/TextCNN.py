# ignore warnings
import warnings

warnings.filterwarnings(action="ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchtext.data import Field, TabularDataset, Iterator
from konlpy.tag import Mecab
from tqdm.notebook import tqdm

from typing import List

# define Field
def create_field(max_len: int = 30):
    # tokenizer
    tokenizer = Mecab()
    # document field
    doc_field = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenizer.morphs,
        lower=True,
        batch_first=True,
        pad_first=True,
        fix_length=max_len,
    )
    # label field
    label_field = Field(sequential=False, use_vocab=False, is_target=True)

    return doc_field, label_field


# Model
class TextCNN(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        kernel_sizes: List[int],
        kernel_num: int = 100,
        class_num: int = 1,
    ):
        super(TextCNN, self).__init__()

        self.kerenel_sizes = kernel_sizes
        self.total_kernel_num = kernel_num * len(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (ks, embed_dim)) for ks in kernel_sizes]
        )
        self.linear = nn.Linear(self.total_kernel_num, class_num)

    def forward(self, seq):
        # embedded_seq: [batch, seq_len, embed_dim]
        embedded_seq = self.embedding(seq)
        # embedded_seq: [batch, input_ch=1, width(seq_len), embed_dim]
        embedded_seq = embedded_seq.unsqueeze(1)

        # conv_results: [(batch, output_ch(kernel_num), width)] * len(kernel_sizes)
        conv_results = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.conv_list]

        # pooled_results: [(batch, output_ch), ...] * len(kernel_sizes)
        pooled_results = [
            F.max_pool1d(features, features.shape[2]).squeeze(2) for features in conv_results
        ]

        # output: [batch, output_ch * len(kernel_sizes)]
        output = torch.cat(pooled_results, dim=1)
        output = F.dropout(output, p=0.5)
        output = self.linear(output)  # [batch, class_num]
        return output.squeeze()

    def loss_fn(self, logits, labels):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels.float())
        return loss

    def accuracy(self, logits, labels):
        logits = torch.round(torch.sigmoid(logits))
        corrects = (logits == labels).float().sum()
        acc = corrects / labels.numel()
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        seqs, labels = train_batch
        logits = self.forward(seqs)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        seqs, labels = val_batch
        logits = self.forward(seqs)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)


if __name__ == "__main__":
    # 1. torchtext Field
    max_len = 30
    doc_field, label_field = create_field(max_len)

    # 2. torchtext Dataset
    train_path = "../data/nsmc/ratings_train.txt"
    test_path = "../data/nsmc/ratings_test.txt"

    trainset, testset = TabularDataset.splits(
        path="../data/nsmc/",
        train="ratings_train.txt",
        test="ratings_test.txt",
        format="TSV",
        fields=[("id", None), ("document", doc_field), ("label", label_field)],
        skip_header=True,
    )

    # 3. build vocab
    doc_field.build_vocab(trainset, min_freq=10, max_size=20000)

    # 4. train/valid dataset split
    trainset, validset = trainset.split(split_ratio=0.7)

    # 5. torchtext train/val/test dataloader
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = Iterator(trainset, batch_size=64, shuffle=True, device=device)
    valid_loader = Iterator(validset, batch_size=64, shuffle=True, device=device)
    test_loader = Iterator(testset, batch_size=64, device=device)

    # 6. Training
    vocab_size = len(doc_field.vocab)
    embed_dim = 100
    kernel_sizes = [3, 4, 5]

    model = TextCNN(vocab_size, embed_dim, kernel_sizes)

    trainer = pl.Trainer(gpus=1, max_epochs=5, val_check_interval=0.5)
    trainer.fit(model, train_loader, valid_loader)