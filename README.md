<h1 align="center">
  <sub>Neko Challenge</sub>
  <br>
  ECG representations
</h1>

<!-- TOC -->
  * [Part1: ML pipeline for cardiovascular risk assessment](#part1-ml-pipeline-for-cardiovascular-risk-assessment)
    * [Preliminary Considerations](#preliminary-considerations)
    * [Unsupervised Learning](#unsupervised-learning)
    * [Supervised Learning](#supervised-learning)
  * [Part2: algorithm for unsupervised electrocardiogram interpretation](#part2-algorithm-for-unsupervised-electrocardiogram-interpretation)
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

## Part1: ML pipeline for cardiovascular risk assessment

### Preliminary Considerations

The goal of this repository is to work in the field of ECG in an unsupervised manner.
An interesting characteristic of the ECGs is their continuity through time.

In general, clinical data appear as structured but scattered data. \
Structured because the value they contain can be easily "understood" by a
machine as simple number. \
Scattered because they suffer from many problem of missing values,
missing units, conversion, interpretation...

An image on the other hand is something that is not structured by itself.
A machine struggles to understand what the image is. We have to build
deep learning models that can progressively build patterns in order to
capture the different relevant pieces of information in the image. \
A human or machine annotator may miss classify an image,
but the signal of the image itself remains uncorrupted. \
Unsupervised learning tries to capture the inherent signal that make up
the image.

We can compare the image use case to the ECG timeseries: ECG are images in 1D.
Now let consider the difficulty of recording an ECG:
we need a process and machines. The whole pipeline may suffer from
miss calibration, there may be sensibility issues. \
As we will see later, a model that would understand
the signal itself in an unsupervised manner could help detect problems of
quality in the data itself and thus resolve the problem of calibration for
example.

Another thing to consider is the shape of the ECGs through time.
They tend to shift and are not cleanly centered around 0.
In deep learning models, much
research has been involved in normalization layers, showing the importance
of having defined ranges of centered values.
As such it would be interesting to consider proper preprocessing before
feeding the ECG to any deep learning model.

### Unsupervised Learning

Before proposing a method to train a model to capture the signal of an ECG,
let us figure out the value of such a model. In the following, this model
will be called the encoder.

This encoder can be used to structure a dataset of patients
for whom we have recorded ECGs.
Using an algorithm such as K means clustering, we may create partitions of
ECGs where the different ECGs in the same cluster are expected to share
similarities in regard of the features that have been extracted by the
encoder.
As the ECGs seem relevant to contain information about cardiovascular risk,
it should be possible to find clusters where patients share some
cardiovascular bad condition. We could use a clinical data such as
"total cholesterol" to find these clusters because "total cholesterol"
is known to be correlated with bad cardiovascular condition. \
Then, with the ANOVA method, it could be possible to
estimate whether the average of "total cholesterol" could be
significantly different in some clusters. Some statistical analysis can
be performed to check whether other clinical data have an abnormal
distribution inside the marked clusters. This could indicate other clinical
data that are correlated with bad cardiovascular risk. All this should
be discussed with clinicians to find and have a guess at the different
clinical data to collect.

We can use t-SNE on the `features` extracted by the encoder. Thanks to this method,
it could be possible to visualize the proximity of different ECGs and find outliers.
Then, with cosine similarity distance on the `features`,
we can find other ECGs with the same problem. These outliers may indicate
cardiac risk but also ECG of bad quality.

At some point, we will need clinicians expertise to annotate the different
ECGs. Once more, the encoder may prove useful with methods like active learning
to make suggestions on the next best samples to annotate.
The best samples could represent ECGs for which the
extracted `features` are far from the `features` that have already been
annotated. This process may be long to initialize because it would need
to iterate on the methodology, create user interface and interact a lot
with clinicians. In the long run, those process will enable continuous
monitoring and feedback.

### Supervised Learning

Finally, the encoder will serve as the backbone to train classifiers
of cardiovascular risk.

Also note that in the end it would be preferable to train a model that relies
on the least


## Part2: algorithm for unsupervised electrocardiogram interpretation

## Installation and Setup

### Installation

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

### Setup

Download the [PTB-XL](https://www.physionet.org/content/ptb-xl/1.0.3/)
dataset to end up with a directory
`~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`.

Any other location is also possible but the `--db` parameter will have
to be updated accordingly in further commands.

## Getting Started

Be sure to be in the `neko` conda environment and to have downloaded the PTB-XL
dataset somewhere (see [previous paragraph](#installation)).

Also note that the subdirectory `weights` already contains some weights that
have been trained with the `small` config of the different models
(see paragraph [Training](#training)).

That way, it is already possible to test the [evaluation](#evaluate-the-encoder)
without having to actually [train the models](#train-an-encoder-and-a-decoder)
first.

### Train an Encoder and a Decoder

Let us train the 2 models of our system: an encoder and a decoder.

The encoder is trained to encode a temporal signal into a vector.
The decoder is trained to decode a temporal signal out of a vector.

The encoder model is our backbone. We will typically rely on it to create
value with finetuning for future model applications.
The decoder is the head that helps training the encoder. Plus, it will
enable some interpretability features (see [later](#investigation)).

The 2 models are trained end to end in a generative way.
To launch the training, go to the `src/script` subdirectory and execute the command:

```shell
python train.py --db ~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv --encoder ../../weights/small/encoder.pt --decoder ../../weights/small/decoder.pt --device cuda --model small
```

`--db`: the input dataset of ECGs. \
`--encoder`: the path to save the encoder model to the disk. \
`--decoder`: the path to save the decoder model to the disk. \
`--device`: the device used to train the models (eg: mps, cuda). \
`--model`: the model config to train.

### Evaluate the Encoder

In order to evaluate the quality of the encoder model,
go to the `src/script` subdirectory and execute the command:

```shell
python eval.py --db ~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv --encoder ../../weights/small/encoder.pt --decoder ../../weights/small/decoder.pt --device cuda --model small
```

`--db`: the input dataset of ECGs. \
`--encoder`: the path to load the encoder model from the disk. \
`--decoder`: the path to load the decoder model from the disk. \
`--device`: the device used to run the models (eg: mps, cuda). \
`--model`: the model config to run.

The command will encode the style of the input ECGs and then decode the style
in order to generate an ECG.
Ideally the generated ECG should be the same as the original one,
more on that [later](#generation-from-style).

## Investigation

First, let us test the `example_physionet.py` file in order to look at the shape of
the 12-lead electrocardiograms:
- 12 different timeseries that are sampled at 100Hz or 500Hz
- in the following, let us focus on the 100Hz ECGs.

In order to create patients' partitions, the plan is the following:
1. for each patient, encode the 12-lead timeseries into features vectors
2. run K means clustering on the previous vectors

The principal difficulty seems to encode the features vectors.
The triplet loss could be a solution:
1. sample an anchor features' vector from patient A,
a positive features' vector from the same patient A
and a negative features' vector from patient B
2. learn that the anchor and positive features should be similar while
the anchor and the negative features should not.

Let us consider
[styleGAN](https://cv-tricks.com/how-to/understanding-stylegan-for-image-generation-using-deep-learning/amp/).
The 12-lead timeseries of the ECG could be modulated by the features vector previously mentioned.
This seems relevant because:
1. a generative learning approach could train the models more effectively than GANs
2. the system would be interpretable: the features' vector (the style)
could be reversed into timeseries that are in the domain of clinicians
3. if the features' vector can be reversed / decoded, it is also possible to explore
small modifications of the style

In order to implement this idea, let us figure out a way to make the style vector
distill information to the different layers of the decoder. This is
already what the `Transformers` do in the `Attention` layers.
Thus, the style vector may be concatenated before the start of
the `sequential` axis and the model will learn how to use it the best way.

Another problem is to limit memory consumption during the training.
In order to do so, let us split the different timeseries in 10 chunks of 1s each.
This is because the computation of the attention `scores`
makes memory grow o(N^2) of `sequential` dimension.

## Results

### Training

The small model config has been trained for 4 epochs.
It lasted around 2 hours on a MacBook Pro with M3Max (64Gb of RAM).
It took around 3.4s per batch which seems a lot.\
A batch is composed of 32 ECGs of 10s, split 10 times, hence a virtual
batch size of 320 with gradient accumulation.\
An epoch is composed of 409 batches (4090 virtual batches with the previous remark).
The loss came from 1.816 to 0.056 but the model could converge more.

Here is some logging corresponding to the previous training.

<figure>
<img src="data/in/small_4epochs.png">
<figcaption>Training small encoder and decoder for 4 epochs</figcaption>
</figure>

Below is a summary of the different trainings:

| Models config | GPU           | nb epochs | batch size | time per batch | final MSE    |
|---------------|---------------|-----------|------------|----------------|--------------|
| small         | Apple's M3Max | 4         | 32 (x10)   | 3.4s           | 1.816->0.056 |
| large         | AMD's 6900XT  | 12        | 32 (x10)   | 4.7s           | 0.028        |
| small         | NVIDIA's T4   | 0         | 32 (x10)   | ~8s            | ?            |

In particular, it is interesting to note that the large model took more epochs
to reach same loss level as the small one (8 epochs against 4 epochs for the small one).
An idea could be to test training the large config with
[LoRA](https://medium.com/@Shrishml/lora-low-rank-adaptation-from-the-first-principle-7e1adec71541).

### Some Sanity Checks

When the style's vector contains only zeroes, the decoder generates a zero
curve which is a good thing:

<figure>
    <img src="data/in/zero.png">
    <figcaption>
    1/12 timeseries of an ECG sampled at 100Hz (the picture is 1s)
    </figcaption>
</figure>

Given some style vector repeated in a batch, the generated curves of the 12-lead are
also repeated (but each timeseries inside the 12-lead is different).

### Generation from Style

Let us look at some curves that have been generated by the encoder and the decoder.

<table align="center" cellspacing="0" cellpadding="0">
    <tr>
        <td><img src="data/in/pat4_lead1.png"></td>
        <td><img src="data/in/pat4_lead1_truth.png"></td>
    </tr>
    <tr>
        <td>Generated curve for patient A lead 1</td>
        <td>Ground truth for patient A lead 1</td>
    </tr>
    <tr>
        <td><img src="data/in/pat3_lead1.png"></td>
        <td><img src="data/in/pat3_lead1_truth.png"></td>
    </tr>
    <tr>
        <td>Generated curve for patient B lead 1</td>
        <td>Ground truth for patient B lead 1</td>
    </tr>
    <tr>
        <td><img src="data/in/pat3_lead3.png"></td>
        <td><img src="data/in/pat3_lead3_truth.png"></td>
    </tr>
    <tr>
        <td>Generated curve for patient B lead 3</td>
        <td>Ground truth for patient B lead 3</td>
    </tr>
</table>

Let us begin with some sanity checks:
- the generated curve for patient A lead 1
is different from the one of patient B lead 1
- the generated curve for patient B lead 1
is different from the one of patient B lead 3

Then, we may compare the generated curves to the ground truth
and see that they are not the same.

There may be many reasons about this situation. \
First, we could consider training the models more, tune hyperparameters...\
Then, we could consider having a bigger decoder.\
But we should also consider the very structure of our current encoder.\
The encoder is composed of `Convolutions` of small kernels. These kernels
may capture patterns that arise at different timings along the `sequential` axis.

Hence, in the curves above, we may recognize similar elements between
the ground truth and the generated curves at different timings.\
Thus, we could try and enhance the encoder
by having `Convolutions` with bigger kernels.

As a conclusion, the interpretable methodology put in place seems promising
in that it gives clues of what patterns the encoder is able to capture or not.

## Next Steps

1. test a new architecture for the encoder
   1. increase the filters of the `Convolutions`
   2. modify the `avgpool`, maybe just `flatten`

2. run K means clustering
   1. keep the current 10x split of the timeseries
   2. check if the different chunks of same patient are in the same cluster
   3. do it for other ECGs of same patient
   4. reverse / decode the different mean vectors (generated by K means) into ECGs
   5. compare with the real ECGs of different patients inside the same cluster
   6. compare the generated mean ECGs that are close from each other
   7. compare the generated mean ECGs that are far from each other

3. train a classifier of one clinical data:
   1. use the `diagnostic_class` given in `example_physionet.py`
   2. try to balance the number of patients for each class or use a weighted loss
   3. take the features' vector as input
   4. compute metric: ROC AUC but also PR AUC or F1 score to take into account imbalanced dataset
   5. the metric serves as a proxy to estimate the quality of the encoder
