# Semantic Textual Similarity

- tf-idf

```shell
$ python tdidf.py
```


- simple fine-tuning

```shell
$ python train.py configs/simple-fine-tuning.yaml
```

- masked language modeling

```shell
$ python train.py configs/only-mlm.yaml
```

- scale-invariant fine-tuning

```shell
$ python train.py configs/simple-sift.yaml
```

- sift after pretraining

```shell
$ python train.py configs/sift-after-tapt.yaml
```

- inference

```shell
$ python inference.py
```
