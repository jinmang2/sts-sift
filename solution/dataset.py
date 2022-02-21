import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


def read_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    return train, test


def get_input_ids(train: pd.DataFrame, cls_token_id: int, sep_token_id: int) -> pd.Series:
    cls_token_ids = pd.Series([[cls_token_id]] * len(train))
    input_ids1 = train["sentence1"].map(str.split).map(lambda x: [int(i) for i in x])
    sep_token_ids = pd.Series([[sep_token_id]] * len(train))
    input_ids2 = train["sentence2"].map(str.split).map(lambda x: [int(i) for i in x])

    input_ids = cls_token_ids + input_ids1 + sep_token_ids + input_ids2
    return input_ids


def get_simple_statistics(train: pd.DataFrame, test: pd.DataFrame, return_outputs: bool = False):
    from collections import Counter

    def _get_minmax_token_id_and_sent_length(df, col_name):
        max_token_id = max(int(i) for l in df[col_name].map(lambda x: x.split()).values for i in l)
        min_token_id = min(int(i) for l in df[col_name].map(lambda x: x.split()).values for i in l)
        max_len_sent = df[col_name].map(lambda x: x.split()).map(len).max()
        min_len_sent = df[col_name].map(lambda x: x.split()).map(len).min()
        return (max_token_id, min_token_id, max_len_sent, min_len_sent)

    max_token_id, min_token_id, max_len_sent, min_len_sent = -1, -1, -1, -1
    for df in (train, test):
        for col in ("sentence1", "sentence2"):
            outputs = _get_minmax_token_id_and_sent_length(df, col)
            _max_token_id, _min_token_id, _max_len_sent, _min_len_sent = outputs
            max_token_id = max(max_token_id, _max_token_id)
            min_token_id = min(min_token_id, _min_token_id)
            max_len_sent = max(max_len_sent, _max_len_sent)
            min_len_sent = min(min_len_sent, _min_len_sent)

    total_tokens = [int(i) for l in train["sentence1"].map(str.split).values for i in l] + \
                   [int(i) for l in train["sentence2"].map(str.split).values for i in l] + \
                   [int(i) for l in test["sentence1"].map(str.split).values for i in l] + \
                   [int(i) for l in test["sentence2"].map(str.split).values for i in l]

    tokens_cnt = sorted(Counter(total_tokens).items())[:5] + ["..."] + sorted(Counter(total_tokens).items())[-5:]

    print(
        "[Analysis results]\n"
        f"{'=' * 50}\n"
        f"train_shape: {train.shape}, test_shape: {test.shape}\n"
        f"train_column_names: {list(train.columns)}\n"
        f"test_column_names: {list(test.columns)}\n"
        f"train_label_value_counts: {dict(train.label.value_counts())}\n"
        f"minimum value of token_ids(min_token_id): {min_token_id}\n"
        f"maximum value of token_ids(max_token_id: {max_token_id}\n"
        f"minimum value of sentence length(min_len_sent): {min_len_sent}\n"
        f"maximum value of sentence length(max_len_sent): {max_len_sent}\n"
        f"tokens_cnt: {tokens_cnt}\n"
    )
    print("2~6까지의 token은 tokenizing 결과로 등장하지 않습니다.")
    print("때문에 위 자리는 미공개 Vocab의 special token id일 가능성이 매우 높습니다.")
    print("본 모델을 학습시킴에 있어 cls, sep, pad, unk, mask token을 위 자리의 토큰으로 채우겠습니다.")

    if return_outputs:
        outputs = {
            "train_shape": train.shape,
            "test_shape": test.shape,
            "train_column_names": list(train.columns),
            "test_column_names": list(test.columns),
            "train_label_value_counts": dict(train.label.value_counts()),
            "min_token_id": min_token_id,
            "max_token_id": max_token_id,
            "min_len_sent": min_len_sent,
            "max_len_sent": max_len_sent,
            "tokens_cnt": sorted(Counter(total_tokens).items()),
        }
        return outputs


class STSDataset(Dataset):

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if y is not None:
            assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        outputs = {"input_ids": self.X[idx]}
        if self.y is not None:
            outputs.update({"labels": self.y[idx]})
        return outputs
