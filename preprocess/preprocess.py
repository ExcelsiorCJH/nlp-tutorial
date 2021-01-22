import pickle
import pandas as pd


def preprocess(data_path: str, save_path: str = None, data_type: str = "nsmc"):
    if data_type == "nsmc":
        data = pd.read_csv(data_path, sep="\t")
        docs = data["document"].tolist()

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(docs, f)
        return None

    return docs


if __name__ == "__main__":
    data_path = "../data/nsmc/ratings_train.txt"
    save_path = "../data/nsmc/nsmc_data.pkl"
    docs = preprocess(data_path=data_path, save_path=save_path)
