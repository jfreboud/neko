<h1 align="center">
  <sub>Neko Challenge</sub>
  <br>
  ECG representations
</h1>

<!-- TOC -->
  * [Installation](#installation)
  * [Getting started](#getting-started)
<!-- TOC -->

## Installation

[![](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)

The following instructions are useful to install the `neko` library in a python environment.
Currently, `neko` supports Python 3.9.

Create a virtual environment and install `poetry`:

```shell
conda create --name neko python=3.9
conda activate neko
pip install poetry
```

Go to the `src` subdirectory and install the `neko` library:

````shell
poetry install
````

## Getting started

### Train an Encoder and a Decoder

First, train 2 models: an encoder and a decoder.
The encoder is trained to encode a temporal signal into a vector.
The decoder is trained to decode a temporal signal out of a vector.

The 2 systems are trained end to end in a generative way.
To launch the training, go to the `script` subdirectory and execute the command:

```shell
python train.py --db ~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv --encoder ../../weights/encoder.pt --decoder ../../weights/decoder.pt --device mps --model small
```

`--db`: the input dataset of ECGs. \
`--encoder`: the path to save the encoder model to the disk. \
`--decoder`: the path to save the decoder model to the disk. \
`--device`: the device used to train the models (I tested mps on MacOS or cuda elsewhere). \
`--model`: the model config to train.

### Evaluate the 2 Models

In order to evaluate the quality of the 2 models,
go to the `script` subdirectory and execute the command:

```shell
python eval.py --db ~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv --encoder ../../weights/encoder.pt --decoder ../../weights/decoder.pt --device mps --model small
```

`--db`: the input dataset of ECGs. \
`--encoder`: the path to load the encoder model from the disk. \
`--decoder`: the path to load the decoder model from the disk. \
`--device`: the device used to run the models (I tested mps on MacOS or cuda elsewhere). \
`--model`: the model config to run.

## Personal Wandering

I began testing the `example_physionet.py` file in order to look at the shape of
the 12-lead electrocardiograms:
- 12 different timeseries that are sampled at 100Hz or 500Hz
- in the following I focus on the 100Hz ECGs.

In order to create patients' partitions, my plan was the following:
1. for each patient, encode the 12-lead timeseries into a vector
2. run the K means clustering on the previous vectors

The principal difficulty seemed to encode the features vector.
I thought about using the triplet loss:
1. sample an anchor features' vector from patient A,
a positive features' vector from the same patient A
and a negative features' vector from patient B
2. learn that the anchor and positive features should be similar while
the anchor and the negative features should not.

I also thought about
[styleGAN](https://cv-tricks.com/how-to/understanding-stylegan-for-image-generation-using-deep-learning/amp/).
The 12 timeseries of the ECG could be modulated by the features vector previously mentioned.
This seemed relevant because:
1. a generative learning approach could train the models more effectively compared to GAN
2. build interpretability by design: the features' vector (the style)
can be decoded into the original timeseries
3. once the features' vector can be decoded, we may explore the modifications of the style

In order to implement this idea, I had to figure out a way to make the style vector
distill information to the different layers of the decoder. But this is
already what the Transformers do in their Attention layers: the style vector may
only be concatenated before the beginning of the `seq` dimension of the timeseries
and the model will learn how to use it thanks to the queries and keys of the Attention layers.

I have tried to limit memory consumption during the training.
In order to do so, I split the different timeseries in 10 chunks of 1s each.
This is because the computation of the attention scores in the Transformers
make memory grow in o(N^2) of `seq` dimension.

## Results

### Training

<figure>
<img src="data/in/small_4epochs.png">
<figcaption>Training small encoder and decoder for 4 epochs</figcaption>
</figure>

### Some Sanity Checks

When the style's vector contains only zeroes, the decoder generates a zero
curve:

<figure>
<img src="data/in/zero.png">
<figcaption>1/12 timeseries of an ECG sampled at 100Hz (the picture is 1s)</figcaption>
</figure>

Given some style vector repeated in a batch, the generated curves of the 12-lead are
also repeated (but each timeseries inside the 12-lead is different).

### Generation from Style

<figure float="left">
  <img src="data/in/pat4_lead1.png" width="150" />
  <img src="data/in/pat3_lead1.png" width="150" />
  <figcaption>Same lead but different patient</figcaption>

  <img src="data/in/pat4_lead1_truth.png" width="150" />
  <img src="data/in/pat3_lead1_truth.png" width="150" />
  <figcaption>Ground truth</figcaption>
</figure>

<br>

<figure float="left">
  <img src="data/in/pat3_lead1.png" width="150" />
  <img src="data/in/pat3_lead3.png" width="150" />
  <figcaption>Same patient but different lead</figcaption>

  <img src="data/in/pat3_lead1_truth.png" width="150" />
  <img src="data/in/pat3_lead3_truth.png" width="150" />
  <figcaption>Ground truth</figcaption>
</figure>

## Next Steps

1. implement K means clustering
   1. keep the split of the timeseries
   2. check if the different chunks of same patient are in the same cluster

2. train a classifier of one clinical data:
   1. use the `diagnostic_class` given in `example_physionet.py`
   2. try to balance the number of patients for each class or use a weighted loss
   3. take the features' vector as input
   4. compute metric: ROC AUC but also PR AUC or F1 score to take into account imbalanced dataset
   5. this metric will serve as a proxy to estimate the quality encoder
