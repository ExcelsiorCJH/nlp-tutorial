import pickle

from collections import Counter
from konlpy.tag import Mecab
from tqdm import tqdm


def preprocess(
    data_path: str,
    word_index: dict = None,
    num_words: int = 10000,
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


def create_contexts_target(corpus, window=3):
    contexts, targets = [], []

    for tokens in tqdm(corpus):
        if len(tokens) > window:
            idx = 0
            while window + idx + 1 <= len(tokens):
                target = tokens[idx + window]
                if target != 0:
                    contexts.append(tokens[idx : idx + window])
                    targets.append(target)

                idx += 1

    return contexts, targets