import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

from solution import set_seed
from solution.args import HfArgumentParser, ModelTrainingArguments
from solution.dataset import read_data, get_input_ids, get_simple_statistics, STSDataset
from solution.trainers import create_model, get_trainer, free_memory


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def fine_tuning(args: ModelTrainingArguments, input_ids: np.ndarray, labels: np.ndarray):
    """ Fine-tuning """
    # Split data to k-fold
    output_dir = args.output_dir
    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    for k, (train_idx, eval_idx) in enumerate(skf.split(input_ids, labels)):
        args.output_dir = os.path.join(output_dir, f"{k}-fold")
        # Get train and eval dataset
        train_dataset = STSDataset(input_ids[train_idx], labels[train_idx])
        eval_dataset = STSDataset(input_ids[eval_idx], labels[eval_idx])
        # Create model from scratch or Load from pre-trained model path
        model = create_model(args)
        # Get trainer with SIFT module (If `apply_sift` is True, this module use sift module.)
        trainer = get_trainer(args, model, train_dataset, eval_dataset)
        trainer.train()
        # After training, free memory (cpu, gpu, garbage collection)
        free_memory(trainer)
        del trainer


def pre_training(args: ModelTrainingArguments, input_ids: np.ndarray) -> str:
    """ Pre-training using masked language modeling """
    train_dataset = STSDataset(input_ids)
    model = create_model(args)
    apply_sift = args.apply_sift
    args.apply_sift = False
    trainer = get_trainer(args, model, train_dataset, None)
    trainer.train()
    tapt_model_path = os.path.join(args.output_dir, "best-tapt")
    model.save_pretrained(tapt_model_path)
    free_memory(trainer)
    args.apply_sift = apply_sift
    return tapt_model_path


# @TODO: 🤗 TrainingArguments num_train_epoch problems
# def tapt_fine_tuning(args: ModelTrainingArguments, input_ids: np.ndarray, labels: np.ndarray):
#     """ Task-Adaptive pre-training with fine-tuning """
#     tapt_model_path = pre_training(args, input_ids)
#     args.training_mode = "sts"
#     args.model_path = tapt_model_path
#     fine_tuning(args, input_ids, labels)


def main(args: ModelTrainingArguments):
    # set seed for reproducibility
    set_seed(args.seed)
    # read data from path
    train, test = read_data(args.data_path)
    # get input_ids and labels
    input_ids = get_input_ids(train, args.cls_token_id, args.sep_token_id).values
    labels = train["label"].values

    if args.training_mode == "sts":
        fine_tuning(args, input_ids, labels)
    elif args.training_mode == "only-mlm":
        args.training_mode = "mlm"
        pre_training(args, input_ids)
    # elif args.training_mode == "tapt":
    #     args.training_mode = "mlm"
    #     tapt_fine_tuning(args, input_ids, labels)

    print("Done.")


if __name__ == "__main__":
    # argument parsing
    parser = HfArgumentParser(ModelTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args, = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()

    assert isinstance(args.k_fold, int) and args.k_fold >= 2, (
        f"k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={args.k_fold}."
    )

    # assert args.training_mode in ["sts", "only-mlm", "tapt"], (
    #     "This module only supports to `sts`, `only-mlm`, and `tapt`."
    # )
    assert args.training_mode in ["sts", "only-mlm"], (
        "This module only supports to `sts`, `only-mlm`."
    )

    os.environ["WANDB_PROJECT"] = args.wandb_project

    main(args)
