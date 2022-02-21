from argparse import Namespace, ArgumentParser
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from solution.dataset import read_data


def main(args: Namespace):
    train, test = read_data(args.data_path)

    X = train[["sentence1", "sentence2"]].values
    y = train["label"].values

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=args.shuffle,
        random_state=args.seed
    )

    for thresh in np.arange(-0.1, 1.1, 0.1):
        total_acc = 0
        print("THRESHOLD:", round(thresh, 1))
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            acc, n = 0, 0
            vectorizer = TfidfVectorizer()
            sentences = [l for line in X[train_idx].tolist() for l in line]
            vectorizer.fit(sentences)
            s1 = vectorizer.transform(X[test_idx][:, 0])
            s2 = vectorizer.transform(X[test_idx][:, 1])
            similarity = s1.dot(s2.T)
            preds = np.where(similarity.diagonal() > thresh, 1, 0)
            for pred, target in zip(preds, y[test_idx]):
                acc += (pred == target)
                n += 1
            pred_0 = (preds == 0).sum()
            pred_1 = (preds == 1).sum()
            print(f"{k}-fold: {acc / n: .4f}, acc: {acc: 4d}, n: {n: 4d}, incorrect: {pred_0: 4d}, correct: {pred_1: 4d}")
            total_acc += acc
        print(f"TOTAL ACCURACY: {total_acc / train.shape[0]: .4f}")
        print("=" * 70)


if __name__ == "__main__":
    # argument parsing
    parser = ArgumentParser(description="Calc similarity using tf-idf.")
    parser.add_argument("--data_path", default=".")
    parser.add_argument("--n_splits", default=5)
    parser.add_argument("--shuffle", default=False)
    parser.add_argument("--seed", default=42)
    args = parser.parse_args()
    if not args.shuffle:
        args.seed = None
    main(args)
