<h1 align="center">
  <sub>Neko Challenge</sub>
  <br>
  ECG representations
</h1>

<!-- TOC -->
  * [Installation and Setup](#installation-and-setup)
    * [Installation](#installation)
    * [Setup](#setup)
  * [Getting Started](#getting-started)
    * [Train an Encoder and a Decoder](#train-an-encoder-and-a-decoder)
    * [Evaluate the Encoder](#evaluate-the-encoder)
  * [Investigation](#investigation)
  * [Results](#results)
    * [Training](#training)
    * [Some Sanity Checks](#some-sanity-checks)
    * [Generation from Style](#generation-from-style)
  * [Next Steps](#next-steps)
<!-- TOC -->

The goal of this repository is to work on the field of ECG in an unsupervised manner.
In a way, we want to summarize / encode the content of timeline series.
From there it will be possible to:

1. explore a dataset of ECGs effectively
   1. create partitions of patients
   2. find correlations with other clinical signals
   3. one question could be:
are the patients in the same cluster frequently associated with overweight?

2. mixing with clinician expertise, labels could help in building
an AI classifier on top of the previous unsupervised representations
   1. impact on patient: predict cardio vascular condition

3. loop back to enhance the quality of data (also with clinician labels)
   1. train a model to detect ECGs of good quality
   2. challenge the current data: discard bad samples
   3. data creation routine:
ask for a new ECG creation as soon as possible when quality is bad

## Installation and Setup

### Installation

[![](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)

The following instructions are useful to install the `neko` library in a python environment.
Currently, `neko` supports Python 3.9 and above (but I did not check more recent versions).

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

### Setup

Download the [PTB-XL](https://www.physionet.org/content/ptb-xl/1.0.3/)
dataset to end up with a directory
`~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`.

Any other location is also possible but the `--db` parameter will have
to be updated in further commands.

## Getting Started

Be sure to be in the `neko` conda environment and to have downloaded the PTB-XL
dataset somewhere (see [previous paragraph](#installation)).

Also note that the subdirectory weights already contains some weights that
have been trained with the `small` config of the different models
(see paragraph [Training](#training)).

That way, it is already possible to test the [Evaluate the Encoder](#evaluate-the-encoder)
paragraph without having to actually [Train an Encoder and a Decoder](#train-an-encoder-and-a-decoder)
first.

### Train an Encoder and a Decoder

First, we train 2 models: an encoder and a decoder.
The encoder is trained to encode a temporal signal into a vector.
The decoder is trained to decode a temporal signal out of a vector.

The encoder model is the important model.
The decoder is the "head" that helps training the encoder. Plus, it will
enable some interpretability features (see [later](#personal-wandering)).

The 2 models are trained end to end in a generative way.
To launch the training, go to the `src/script` subdirectory and execute the command:

```shell
python train.py --db ~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv --encoder ../../weights/encoder.pt --decoder ../../weights/decoder.pt --device mps --model small
```

`--db`: the input dataset of ECGs. \
`--encoder`: the path to save the encoder model to the disk. \
`--decoder`: the path to save the decoder model to the disk. \
`--device`: the device used to train the models (I tested mps on MacOS). \
`--model`: the model config to train.

### Evaluate the Encoder

In order to evaluate the quality of the encoder model,
go to the `src/script` subdirectory and execute the command:

```shell
python eval.py --db ~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv --encoder ../../weights/encoder.pt --decoder ../../weights/decoder.pt --device mps --model small
```

`--db`: the input dataset of ECGs. \
`--encoder`: the path to load the encoder model from the disk. \
`--decoder`: the path to load the decoder model from the disk. \
`--device`: the device used to run the models (I tested mps on MacOS). \
`--model`: the model config to run.

The command will encode the style of the input ECGs and then decode the style
in order to generate an ECG.
Ideally the generated ECG should be the same as the original one.

## Investigation

I began testing the `example_physionet.py` file in order to look at the shape of
the 12-lead electrocardiograms:
- 12 different timeseries that are sampled at 100Hz or 500Hz
- in the following I focus on the 100Hz ECGs.

In order to create patients' partitions, my plan was the following:
1. for each patient, encode the 12-lead timeseries into features vectors
2. run the K means clustering on the previous vectors

The principal difficulty seemed to encode the features vectors.
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
2. the system would be interpretable by design: the features' vector (the style)
can be decoded into the original timeseries
3. once the features' vector can be decoded, we may explore small modifications of the style

In order to implement this idea, I had to figure out a way to make the style vector
distill information to the different layers of the decoder. But this is
already what the `Transformers` do in their `Attention` layers: the style vector may
only be concatenated before the beginning of the `seq` dimension
and the model will learn how to use it thanks to the `queries` and `keys` of the `Attention` layers.

I have tried to limit memory consumption during the training.
In order to do so, I split the different timeseries in 10 chunks of 1s each.
This is because the computation of the attention `scores` in the `Transformers`
makes memory grow along o(N^2) of `seq` dimension.

## Results

### Training

The small model config has been trained for 4 epochs.
It lasted around 2 hours on a MacBook with M3Max.
The loss came from 1.816 to 0.056 but did not converge yet.

<figure>
<img src="data/in/small_4epochs.png">
<figcaption>Training small encoder and decoder for 4 epochs</figcaption>
</figure>

The large model config has been trained for 12 epochs.
Its final loss was 0.028, this model did not converge either.

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

Here are some curves that have been generated by the encoder and the decoder.
We may compare them to the ground truth.

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

We can see that the generated curves are not the same as the ground truth.

This may come from the structure of the encoder.
The encoder is composed of `Convolutions` of small kernels. These kernels
may capture patterns that arise at different timings inside the `seq` dimension.

Hence, in the curves above, we can recognize similar elements between
the ground truth seem and the generated curves.

In order to better preserve the shape of the timeseries, we should rework the
encoder with less moving parts.

As a conclusion, the methodology put in place seems very interesting
to give an idea of what the encoder is able to capture or not.

## Next Steps

1. test a new architecture for the encoder
   1. try to increase the filters of the `Convolutions`
   2. modify the `avgpool`

2. implement K means clustering
   1. keep the split of the timeseries
   2. check if the different chunks of same patient are in the same cluster

3. train a classifier of one clinical data:
   1. use the `diagnostic_class` given in `example_physionet.py`
   2. try to balance the number of patients for each class or use a weighted loss
   3. take the features' vector as input
   4. compute metric: ROC AUC but also PR AUC or F1 score to take into account imbalanced dataset
   5. this metric will serve as a proxy to estimate the quality encoder
