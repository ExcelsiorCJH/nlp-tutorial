import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from typing import List
from utils import preprocess, create_contexts_target


# collate function for dataloader
def collate_fn(batch):
    tokens = [entry[0] for entry in batch]
    targets = [entry[1] for entry in batch]

    tokens = torch.LongTensor(tokens)
    targets = torch.LongTensor(targets)

    return tokens, targets


# DataSet
class NSMCDataset(Dataset):
    def __init__(self, contexts: List[List[int]], targets: List[int]):
        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


# Model
class NPLM(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        window: int,
        embed_dim: int = 100,
        hidden_dim: int = 50,
    ):
        super(NPLM, self).__init__()

        self.C = nn.Embedding(vocab_size, embed_dim)
        self.H = nn.Linear(window * embed_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.W = nn.Linear(window * embed_dim, vocab_size)

    def forward(self, x):
        x = self.C(x)  # [batch_size, window, embed_dim]
        x = x.reshape(-1, x.shape[1] * x.shape[2])  # [batch_size, window * embed_dim]
        tanh = torch.tanh(self.H(x))  # [batch_size, hidden_dim]
        output = self.W(x) + self.U(tanh)  # [batch_size, vocab_size]
        return output

    def loss_fn(self, logits, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)


if __name__ == "__main__":
    # 1. preprocess
    train_path = "../data/nsmc/train_data.pkl"
    test_path = "../data/nsmc/test_data.pkl"

    train_corpus, word_index, index_word = preprocess(train_path)
    test_corpus, _, _ = preprocess(test_path, word_index)

    train_contexts, train_targets = create_contexts_target(train_corpus)
    test_contexts, test_targets = create_contexts_target(test_corpus)

    # 2. DataSet & DataLoader
    trainset = NSMCDataset(train_contexts, train_targets)
    testset = NSMCDataset(test_contexts, test_targets)

    train_loader = DataLoader(
        dataset=trainset, batch_size=256, collate_fn=collate_fn, shuffle=True, num_workers=8
    )

    test_loader = DataLoader(
        dataset=testset,
        batch_size=256,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=8,
    )

    # 3. Training
    vocab_size = len(word_index)
    window = 3

    model = NPLM(vocab_size, window)

    trainer = pl.Trainer(gpus=2, max_epochs=10, val_check_interval=0.5, accelerator="dp")
    trainer.fit(model, train_loader, test_loader)