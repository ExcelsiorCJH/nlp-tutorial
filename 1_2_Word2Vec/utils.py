import pickle
import random

from collections import Counter
from konlpy.tag import Mecab
from tqdm import tqdm
from typing import List


def preprocess(
    data_path: str, word_index: dict = None, num_words: int = 10000,
):
    tokenizer = Mecab()

    # 0. data load
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # 1. bag-of-words
    vocab, docs = [], []
    for doc in tqdm(data):
        if doc:
            # nsmc 데이터에 nan값을 제외해주기 위함
            try:
                nouns = tokenizer.nouns(doc)
                vocab.extend(nouns)
                docs.append(nouns)
            except:
                continue

    # 2. build vocab
    if not word_index:
        vocab = Counter(vocab)
        vocab = vocab.most_common(num_words)

        # 3. add unknwon token
        word_index = {"<UNK>": 0}
        for idx, (word, _) in enumerate(vocab, 1):
            word_index[word] = idx

    index_word = {idx: word for word, idx in word_index.items()}

    # 4. create corpus
    corpus = []
    for doc in docs:
        if doc:
            corpus.append([word_index.get(word, 0) for word in doc])

    return corpus, word_index, index_word


# Reference:
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py
def skipgrams(
    sequence: List[int], vocab_size: int, window_size: int = 4, negative_samples: int = 1.0,
):
    couples, labels = [], []
    for i, wi in enumerate(sequence):
        if not wi:  # <UNK> 토큰일 경우
            continue
        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        #     print(window_start, window_end)

        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:  # <UNK> 토큰일 경우
                    continue

                couples.append([wi, wj])
                labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [
            [words[idx % len(words)], random.randint(1, vocab_size - 1)]
            for idx in range(num_negative_samples)
        ]

        labels += [0] * num_negative_samples

    return couples, labels
