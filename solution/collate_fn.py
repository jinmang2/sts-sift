from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


DictListInputs = Dict[str, List[int]]
DictTensorInputs = Dict[str, torch.Tensor]


class PadDataCollator:

    def __init__(self, args):
        self.pad_token_id = args.pad_token_id

    def __call__(self, x: List[DictListInputs]) -> DictTensorInputs:
        outputs = self.pad(x)
        batch_encoding = {"input_ids": outputs[0]}
        if len(outputs) > 1:
            batch_encoding.update({"labels": outputs[1]})
        return batch_encoding

    def pad(self, x: List[DictListInputs]) -> Tuple[torch.Tensor, ...]:
        X = [inp["input_ids"] for inp in x]

        use_label = False
        if x[0].get("labels", None) is not None:
            use_label = True
            y = [inp["labels"] for inp in x]

        max_len = max(len(sent) for sent in X)
        padded = [sent + [self.pad_token_id] * (max_len - len(sent)) for sent in X]
        outputs = (torch.LongTensor(padded),)
        if use_label:
            outputs += (torch.LongTensor(y),)

        return outputs


class MLMDataCollator(PadDataCollator):

    def __init__(self, args):
        super().__init__(args)
        self.vocab_size = args.vocab_size
        self.cls_token_id = args.cls_token_id
        self.sep_token_id = args.sep_token_id
        self.unk_token_id = args.unk_token_id
        self.mask_token_id = args.mask_token_id
        self.mlm_prob = args.mlm_prob

    def __call__(self, x: List[DictListInputs]) -> DictTensorInputs:
        inputs = self.pad(x)
        inputs, labels = self.mask_tokens(inputs)
        return {"input_ids": inputs, "labels": labels}

    @property
    def special_tokens(self) -> List[int]:
        return [self.cls_token_id, self.sep_token_id, self.unk_token_id, self.pad_token_id]

    def get_special_tokens_mask(self, val: List[int]) -> List[int]:
        special_tokens_mask = []
        for v in val:
            special_tokens_mask.append(1 if v in self.special_tokens else 0)
        return special_tokens_mask

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability 0.15)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        padding_mask = labels.eq(self.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with mask_token_id
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
