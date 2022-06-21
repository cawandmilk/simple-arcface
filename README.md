# Simple Implementation of ArcFace

This repository is the official implementation for the Natural Language Processing (GSI7625.01-00, Professor [Ha Young Kim](https://sites.google.com/view/mlcf/cv?authuser=0)) course at Yonsei University in the first semester of 2022. We sincerely appreciate the hard work of the professor.

## Requirements

To install requirements:

```bash
> python -m venv venv
> source ./venv/bin/activate
> pip install -r requirements.txt
```

You need to crawl the necessary data.

```bash
> python prepare.py --data data --max-iter 1000
```

## Training

### Run on Terminal

To train the model in the paper, run this command:

* Baseline

```train
> python train.py \
    --model_name baseline-aug \
    --is_baseline \
    --data data \
    --test_size 0.2 \
    --random_state 42 \
    --buffer_size 30000 \
    --batch_size 32 \
    --lr 0.0003 \
    --epochs 200 \
    --augment \
    --logs logs \
    --ckpt ckpt
```

* ArcFace

```bash
python train.py \
    --model_name arcface-aug \
    --data data \
    --test_size 0.2 \
    --random_state 42 \
    --buffer_size 30000 \
    --batch_size 32 \
    --lr 0.0003 \
    --epochs 200 \
    --augment \
    --logs logs \
    --ckpt ckpt
```

### Run on Jupyter Notebook

If you use the magic command in `Jupyter Notebook`, you can easily iterate through the code and collect the experimental results. =)

```python
## Perform five time iterations.
%run -i -t -N5 train.py --model_name arcface --augment
```

## Results

### Training Logs

All training logs are commited publicaly on [TensorBoard](https://tensorboard.dev/experiment/wjYubhNkRK6doa9F56TaLA/).

### Quantitative Results

Our model achieves the following performance of our own dataset:

| Architecture | Margin | Emb. Dim | Aug.  | Test Loss | Test Acc. (%) |
| :----------- | :----: | :------: | :---: | --------: | ------------: |
| Baseline     | -      | 512      | False | 1.045     | **84.1**      |
| ArcFace      | 20     | 512      | False | 1.119     | 72.8          |
| ArcFace      | 20     | 512      | True  | **0.609** | 81.7          |

## Citation

Please cite below if you make use of the code.

```latex
@misc{oh2022simple,
    title={Simple Implementation of ArcFace},
    author={Myung Gyo Oh},
    year={2022},
    howpublished={\url{https://github.com/cawandmilk/simple_arcface}},
}
```
