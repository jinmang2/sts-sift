from typing import List, Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelTrainingArguments(TrainingArguments):
    data_path: str = field(
        default=".",
        metadata={
            "help": "Data path to use for training and inference."
        }
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model path for `from_pretrained` method."
        }
    )
    wandb_project: str = field(
        default="huggingface",
        metadata={
            "help": "Weights and Biases project name."
        }
    )
    apply_sift: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply Scale-Invariant Fine Tuning(SIFT) or not"
        },
    )
    training_mode: str = field(
        default="sts",
        metadata={
            "help": 'Whether to learn in which mode ["sts", "mlm"].'
        },
    )
    k_fold: int = field(
        default=5,
        metadata={
            "help": "Number of folds"
        }
    )
    vocab_size: int = field(
        default=30522,
        metadata={
            "help": "Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the "
            "`inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`]."
        },
    )
    hidden_size: int = field(
        default=768,
        metadata={
            "help": "Dimensionality of the encoder layers and the pooler layer."
        },
    )
    num_hidden_layers: int = field(
        default=12,
        metadata={
            "help": "Number of hidden layers in the Transformer encoder."
        },
    )
    num_attention_heads: int = field(
        default=12,
        metadata={
            "help": "Number of attention heads for each attention layer in the Transformer encoder."
        },
    )
    intermediate_size: int = field(
        default=3072,
        metadata={
            "help": 'Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.'
        },
    )
    hidden_act: int = field(
        default="gelu",
        metadata={
            "help": 'The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.'
        },
    )
    hidden_dropout_prob: int = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    attention_probs_dropout_prob: int = field(
        default=0.1,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        },
    )
    max_position_embeddings: int = field(
        default=512,
        metadata={
            "help": "The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048)."
        },
    )
    type_vocab_size: int = field(
        default=2,
        metadata={
            "help": "The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`]."
        },
    )
    initializer_range: int = field(
        default=0.02,
        metadata={
            "help": "The standard deviation of the truncated_normal_initializer for initializing all weight matrices."
        },
    )
    layer_norm_eps: int = field(
        default=1e-12,
        metadata={
            "help": "The epsilon used by the layer normalization layers."
        },
    )
    position_embedding_type: int = field(
        default="absolute",
        metadata={
            "help": (
                'Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For'
                'positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to'
                '[Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).'
                'For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models'
                'with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).'
            )
        },
    )
    cls_token_id: int = field(
        default=2,
        metadata={
            "help": "Special token id for [CLS] (classification)."
        },
    )
    sep_token_id: int = field(
        default=3,
        metadata={
            "help": "Special token id for [SEP] (separator)."
        },
    )
    unk_token_id: int = field(
        default=5,
        metadata={
            "help": "Special token id for [UNK] (unknown)."
        },
    )
    pad_token_id: int = field(
        default=4,
        metadata={
            "help": "Special token id for [PAD] (padding)."
        },
    )
    mask_token_id: int = field(
        default=6,
        metadata={
            "help": "Special token id for [MASK] (mask)."
        },
    )
    mlm_prob: float = field(
        default=0.15,
        metadata={
            "help": "Probability for MLM (masked language modeling)."
        },
    )

    @property
    def from_scratch(self):
        return self.model_path is None
