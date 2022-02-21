import gc
import torch
from transformers import Trainer
from transformers import PreTrainedModel
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaForMaskedLM,
)

from .args import ModelTrainingArguments
from .collate_fn import PadDataCollator, MLMDataCollator
from .metrics import compute_sts_metrics
from .sift import hook_sift_layer, AdversarialLearner


class SiftTrainer(Trainer):
    """ 🤗 Trainer with Scale Invariant Fine-Tuning. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_init is None:
            self.set_adv_module(self.model)

    def set_adv_module(self, model: torch.nn.Module):
        hidden_size = self.model.config.hidden_size
        adv_modules = hook_sift_layer(model, hidden_size=hidden_size)
        adv = AdversarialLearner(self.model, adv_modules)
        self.adv = adv

    def call_model_init(self, trial=None): # override
        model = super().call_model_init(trial=trial)
        self.set_adv_module(model)
        return model

    def compute_loss(self, model, inputs, return_outputs=False): # override
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # =======================================================================================
        # @jinmang2 2022.02.21
        # Apply the Scale-Invariant Fine-Tuning module proposed by DeBERTa if training mode
        if model.training:

            def logits_fn(model, *args, **kwargs):
                outputs = model(*args, **kwargs)
                # We don't use .logits here since the model may return tuples instead of ModelOutput.
                return outputs["logits"] if isinstance(outputs, dict) else outputs[1]

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            loss = loss + self.adv.loss(logits, logits_fn, **inputs)
        # =======================================================================================

        return (loss, outputs) if return_outputs else loss


def get_model_class(training_mode: str):
    if training_mode == "sts":
        model_class = RobertaForSequenceClassification
    elif training_mode == "mlm":
        model_class = RobertaForMaskedLM
    else:
        raise NotImplementedError("There are only solutions for `sts` and `mlm`.")
    return model_class


def create_model(args: ModelTrainingArguments) -> PreTrainedModel:

    config = RobertaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=args.initializer_range,
        layer_norm_eps=args.layer_norm_eps,
        pad_token_id=args.pad_token_id,
        position_embedding_type=args.position_embedding_type,
    )

    model_class = get_model_class(args.training_mode)
    if args.from_scratch:
        model = model_class(config)
    else:
        model = model_class.from_pretrained(args.model_path)
    return model


def get_trainer(
    args: ModelTrainingArguments,
    model: PreTrainedModel,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
) -> Trainer:

    trainer_class = SiftTrainer if args.apply_sift else Trainer
    compute_metrics = None if args.training_mode == "mlm" else compute_sts_metrics
    collate_fn = MLMDataCollator(args) if args.training_mode == "mlm" else PadDataCollator(args)

    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    return trainer


def free_memory(trainer: Trainer, device: str = "cpu"):

    def _optimizer_to(self, device: str = "cpu"):
        # https://github.com/pytorch/pytorch/issues/8741
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(
                                device)

    def _scheduler_to(self, device: str = "cpu"):
        # https://github.com/pytorch/pytorch/issues/8741
        for param in self.lr_scheduler.__dict__.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)

    trainer.model.to(device)
    _optimizer_to(trainer, device)
    _scheduler_to(trainer, device)
    torch.cuda.empty_cache()
    gc.collect()
