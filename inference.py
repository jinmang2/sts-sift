from easydict import EasyDict
from argparse import Namespace, ArgumentParser
from typing import Dict, List, Union, Any

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from solution.collate_fn import PadDataCollator
from solution.dataset import read_data, get_input_ids, STSDataset
from solution.trainers import get_model_class


def prepare_inputs(
    inputs: Dict[str, Union[torch.Tensor, Any]],
    device: str = torch.device
) -> Dict[str, Union[torch.Tensor, Any]]:
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


@torch.no_grad()
def predict(
    model: PreTrainedModel,
    test_dataset: STSDataset,
    batch_size: int
) -> np.ndarray:
    data_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_collate_fn)
    arr = np.array([], dtype=np.int64).reshape(0, 2)
    for batch in data_loader:
        batch = prepare_inputs(batch, model.device)
        outputs = model(**batch)
        arr = np.vstack([arr, outputs.logits.detach().cpu().numpy()])
    return arr


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference")
    parser.add_argument("--data_path", default=".")
    parser.add_argument("--cls_token_id", default=2)
    parser.add_argument("--sep_token_id", default=3)
    parser.add_argument("--pad_token_id", default=4)
    parser.add_argument("--best_ckpts_file", default="best_ckpts.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--submission_file", default="submissions.csv")
    args = parser.parse_args()

    train, test = read_data(args.data_path)
    input_ids = get_input_ids(test, cls_token_id=2, sep_token_id=3)
    pad_collate_fn = PadDataCollator(EasyDict(pad_token_id=args.pad_token_id))
    preds = np.zeros((test.shape[0], 2))
    with open(args.best_ckpts_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            best_ckpt = line.strip()
            model = get_model_class("sts").from_pretrained(best_ckpt)
            model.to(args.device)
            test_dataset = STSDataset(input_ids)
            _preds = predict(model, test_dataset, args.batch_size)
            preds = preds + _preds
    y_hats = preds.argmax(-1)
    submissions = pd.DataFrame(test["id"])
    submissions["label"] = y_hats
    submissions.to_csv(args.submission_file, index=False)
