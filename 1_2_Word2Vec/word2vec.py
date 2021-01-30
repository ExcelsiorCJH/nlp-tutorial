import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import preprocess, skipgrams

from typing import List


# collate function for dataloader
def collate_fn(batch):
    targets = [entry[0][0] for entry in batch]
    contexts = [entry[0][1] for entry in batch]
    labels = [entry[1] for entry in batch]

    targets = torch.LongTensor(targets)
    contexts = torch.LongTensor(contexts)
    labels = torch.FloatTensor(labels)

    return targets, contexts, labels


# Dataset
class W2VDataset(Dataset):
    def __init__(self, pairs: List[List[int]], labels: List[int]):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


# Model
class Word2Vec(pl.LightningModule):
    def __init__(self, vocab_size: int, embed_dim: int = 100):
        super(Word2Vec, self).__init__()

        self.input_embed = nn.Embedding(vocab_size, embed_dim)
        self.output_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context):
        u = self.input_embed(target)  # [batch_size, embed_dim]
        v = self.output_embed(context)  # [batch_size, embed_dim]

        score = torch.sum(u * v, dim=1)  # [batch_size]
        return score

    def loss_fn(self, logits, labels):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
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
        targets, contexts, labels = train_batch
        logits = self.forward(targets, contexts)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        targets, contexts, labels = val_batch
        logits = self.forward(targets, contexts)
        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)


if __name__ == "__main__":
    # 1. preprocess
    train_path = "../data/nsmc/train_data.pkl"
    test_path = "../data/nsmc/test_data.pkl"

    train_corpus, word_index, index_word = preprocess(train_path)
    test_corpus, _, _ = preprocess(test_path, word_index)

    # 2. create skipgrams
    # train skipgrams
    vocab_size = len(word_index)
    train_pairs, train_labels = [], []
    for sequence in tqdm(train_corpus):
        pairs, targets = skipgrams(sequence, vocab_size, negative_samples=0.5)
        train_pairs.extend(pairs)
        train_labels.extend(targets)

    # test skipgrams
    vocab_size = len(word_index)
    test_pairs, test_labels = [], []
    for sequence in tqdm(test_corpus):
        pairs, targets = skipgrams(sequence, vocab_size, negative_samples=0.5)
        test_pairs.extend(pairs)
        test_labels.extend(targets)

    # 3. Dataset & DataLoader
    trainset = W2VDataset(train_pairs, train_labels)
    testset = W2VDataset(test_pairs, test_labels)

    train_loader = DataLoader(
        dataset=trainset, batch_size=256, collate_fn=collate_fn, shuffle=True, num_workers=8,
    )

    test_loader = DataLoader(
        dataset=testset, batch_size=256, collate_fn=collate_fn, shuffle=False, num_workers=8,
    )

    # 4. Training
    vocab_size = len(word_index)
    model = Word2Vec(vocab_size)

    trainer = pl.Trainer(gpus=2, max_epochs=10, val_check_interval=0.5, accelerator="dp",)
    trainer.fit(model, train_loader, test_loader)
