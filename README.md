<h1 align="center">
  <sub>Neko Challenge</sub>
  <br>
  ECGs for cardiovascular risk assessment
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
    * [Conclusion](#conclusion)
    * [Next Steps](#next-steps)
<!-- TOC -->

# Part1: ML pipeline for cardiovascular risk assessment

## Preliminary Considerations

The goal of this repository is to work on ECGs in an unsupervised manner.
An interesting characteristic of the ECGs is their continuity through time.

In general, clinical data appear as structured but scattered data. \
Structured because the value they contain can be easily "understood" by a
machine as a simple number. \
Scattered because they suffer from many problems: missing values,
missing units, conversion, interpretation...

An image on the other hand is something that is not structured by itself.
A machine struggles to understand what the image is about. We have to build
deep learning models that can progressively find patterns in order to
capture the relevant pieces of information to structure this data. \
A human or machine annotator may miss classify an image,
but the signal of the image itself remains uncorrupted. \
Unsupervised learning tries to capture the inherent signal that make up
the image.

We can compare ECG timeseries to the image use case: ECGs are images in 1D.
Now let consider the difficulty of recording an ECG:
we need processes and machines. The whole pipeline may suffer from
miss calibration, sensibility issues.

Another thing to consider is the shape of the ECGs through time.
They tend to shift and are not cleanly centered around 0.
In deep learning models, much
research has been involved in normalization layers, showing the importance
of having defined ranges of centered values.
As such, it would be interesting to consider proper preprocessing before
feeding the ECG to any deep learning model.

## Unsupervised Learning

In the [second part](#part2-algorithm-for-unsupervised-electrocardiogram-interpretation),
we will present a method to train a model to capture
the inherent signal of an ECG in an unsupervised way.
This model will be called the encoder.
In the meanwhile, let us figure out the value of such an encoder.

First, the encoder can be used to structure a dataset of patients
from whom we have recorded ECGs. \
Using an algorithm such as K means clustering, we may create partitions of
ECGs where the different ECGs in the same cluster are expected to share
similarities in regard of the `features` that have been extracted by the
encoder.\
As the ECGs seem relevant to contain information about cardiovascular risk,
it should be possible to find clusters where patients share some
cardiovascular bad condition. For that, we could use a clinical biomarker such as
"total cholesterol" because it is known to be correlated
with bad cardiovascular condition. \
Then, with the ANOVA method, it should be possible to
estimate whether the average of "total cholesterol" is
significantly different in some clusters.
Some statistical analysis can then be performed on those clusters of interest
to find other clinical data that have abnormal distributions compared
to the other clusters. These clinical data become of high interest
to mark bad cardiovascular risk. All this should
be discussed with clinicians in order to settle on a set of
clinical data to collect.

Then, we can use t-SNE on the `features` extracted by the encoder. Thanks to this method,
it could be possible to visualize the proximity of different ECGs and find outliers.
With cosine similarity distance on the `features`,
we can find other ECGs that share the same characteristics.
These outliers may indicate cardiac risk but also ECG of bad quality.

At some point, we will need clinicians' expertise to annotate the different
ECGs. The annotations will be about the content: what can I see in the ECG
as a specialist of health. This information will be precious to
[train a supervised classifier](#supervised-learning).
But the annotations should also tackle the quality of the data itself.
This will help to curate the existing database but also to create a routine to
re record ECG as soon as possible. \
Once more, the encoder is useful to foster effective annotation process
with the help of active learning
to make suggestions on the next best samples to annotate.
These best samples could represent ECGs for which the
extracted `features` are far from the `features` that have already been
annotated.

The data quality consideration may be the starting point to initialize
a long term process between clinicians and the data science team:
iterate on the methodology, create user interface, visualize clusters.
This will enable continuous
monitoring and feedback to eventually fuel a solid database of
ECGs. From there it will be possible to loop back to retrain the encoder,
paving the way to a virtuous circle.

## Supervised Learning

Finally, the `features` generated by the encoder will serve as a data modality
for training classifiers of cardiovascular risk.

The clinical data we talked about
in the [previous paragraph](#unsupervised-learning) may be used
as an input as well. \
Still, we should challenge any clinical data that is
fed to the classifier because we already rely on
the tailored extracted `features` of the ECGs.
Adding other clinical data may add noise to the global signal.
Therefore, features importance should help in that area.

# Part2: algorithm for unsupervised electrocardiogram interpretation

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
dataset to end up with a directory such as
`~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`.

Any other location is also possible but the `--db` parameter will have
to be updated accordingly in [future commands](#train-an-encoder-and-a-decoder).

## Getting Started

Be sure to be in the `neko` conda environment and to have downloaded the PTB-XL
dataset somewhere (see [previous paragraph](#installation)).

Also note that the subdirectory `weights` already contains parameters that
have been trained with the `small` config of the different models
(see paragraph [training](#training)).

Hence, it is possible to run the [evaluation](#evaluate-the-encoder)
without having to actually [train the models](#train-an-encoder-and-a-decoder)
first.

### Train an Encoder and a Decoder

Let us train the 2 models of our system: an encoder and a decoder.

The encoder is trained to encode a temporal signal into a vector.\
The decoder is trained to decode a temporal signal out of a vector.

The encoder model is our backbone. We will typically rely on it to create
the value we mentioned in the [first part](#part1-ml-pipeline-for-cardiovascular-risk-assessment).
The decoder is the head that helps in training the encoder. Plus, it will
enable some interpretability features
(see the [investigation paragraph](#investigation)).

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

In order to evaluate the quality of the encoder,
go to the `src/script` subdirectory and execute the command:

```shell
python eval.py --db ~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv --encoder ../../weights/small/encoder.pt --decoder ../../weights/small/decoder.pt --device cuda --model small
```

`--db`: the input dataset of ECGs. \
`--encoder`: the path to load the encoder model from the disk. \
`--decoder`: the path to load the decoder model from the disk. \
`--device`: the device used to run the models (eg: mps, cuda). \
`--model`: the model config to run.

The command will encode the `style`
(we will talk about it in the [investigation paragraph](#investigation))
of the input ECGs and then decode the `style` in order to generate a new ECG.
Ideally the generated ECG should be the same as the original one,
more on that [later](#generation-from-style).

## Investigation

First, let us test the `example_physionet.py` file in order to look at the shape of
the 12-lead electrocardiograms:
- 12 different timeseries that are sampled at 100Hz or 500Hz
- in the following, we will focus on the 100Hz ECGs that are shorter
- the timeseries are not cleanly centered around 0, we will need some preprocessing.

Then, the difficulty is to find a way to train the encoder in an
unsupervised way.
The triplet loss could be a solution:
1. sample anchor `features` from patient A,
positive `features` from another record of same patient A
and negative `features` from patient B
2. learn that the anchor and positive `features` should be similar while
the anchor and the negative `features` should not.

We can also think of
[styleGAN](https://cv-tricks.com/how-to/understanding-stylegan-for-image-generation-using-deep-learning/amp/).
The 12-lead timeseries of the ECG could be modulated by the `features` of the
encoder that would act as the `style` vector.
We could even leverage generative learning and build an interpretable system:
1. a generative learning approach could train the models more effectively than GANs
2. the `features` (the `style` vector)
could be decoded into timeseries that are in the domain of clinicians
3. it would be possible to explore the impact of small modifications of the `style`

In order to implement this idea, we must find a way to make the `style` vector
distill time-aware information to the different layers of the decoder.
In fact, this is
already what the `Transformers` do in the `Attention` layers thanks to the
`queries` and `values`.
Therefore, the `style` vector may be concatenated before the start of
the `sequential` axis and the model will learn how to use it the best way.

Another problem is to limit memory consumption during the training.
In order to do so, let us split the different timeseries in 10 chunks of 1s each.
This is because the computation of the attention `scores`
makes memory grow o(N^2) of the `sequential` dimension.

## Results

### Training

A small model config (referring to a small encoder and decoder version)
has been trained for 4 epochs.\
It lasted around 2 hours on a MacBook Pro with M3Max.\
It took around 3.4s per batch which seems a lot.\
A batch is composed of 32 ECGs of 10s, split 10 times, hence a virtual
batch size of 320 with gradient accumulation.\
An epoch is composed of 409 batches (virtually 4090).\
The loss came from 1.816 to 0.056, the model could converge more.

Here is some logging corresponding to the previous training.

<img src="data/in/small_4epochs.png">

Below is a summary of the different trainings tested:

| Model's config | GPU           | nb epochs | batch size | time per batch | final MSE    |
|----------------|---------------|-----------|------------|----------------|--------------|
| small          | Apple's M3Max | 4         | 32 (x10)   | 3.4s           | 1.816->0.056 |
| large          | AMD's 6900XT  | 12        | 32 (x10)   | 4.7s           | 0.028        |
| small          | NVIDIA's T4   | 0         | 32 (x10)   | ~8s            | ?            |

In particular, it is interesting to note that the large model took more epochs
to reach same loss level as the small one (8 epochs against 4 epochs for the small one).
An idea could be to test training the large model with
[LoRA](https://medium.com/@Shrishml/lora-low-rank-adaptation-from-the-first-principle-7e1adec71541).

### Some Sanity Checks

When the `style`'s vector contains only zeroes, the decoder generates a zero
curve:

<img src="data/in/zero.png">

Given some `style` vector repeated in a batch, the generated curves of the 12-lead are
also repeated and each timeseries inside the 12-lead is different.

### Generation from Style

Let us look at some curves that have been generated by the encoder and the decoder
trained with the large config. \
The decoder generates curves of shape `(B, seq_dim, 12)`. In order
to match the training setting we will use `seq_dim = 10`.\
During the generation process,
the decoder starts with the zero point (`(B, 1, 12)` filled with 0)
and the `style` vector of shape `(B, encoder_hidden_dim)`.\
We can check that the different generated curves below
do start from 0 while the ground truth do not.

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

Some sanity checks:
- the generated curve for patient A lead 1
is different from the one of patient B lead 1
- the generated curve for patient B lead 1
is different from the one of patient B lead 3

Then, we may compare the generated curves to the ground truth
and see that they are not the same.

There may be many reasons about this situation. \
First, we could consider training the models more, tune hyperparameters.\
Then, we could consider having a bigger decoder.\
But we should also consider the very structure of our current encoder.\
The encoder is composed of `Convolutions` of small kernels. These kernels
may capture patterns that arise at different timings along the `sequential` axis.

Hence, in the curves above, we may recognize similar elements between
the ground truth and the generated curves at different timings
which seems promising.\
Thus, we could try to enhance the encoder
by having `Convolutions` with bigger kernels.

## Conclusion

We designed and implemented an original style generative way of training
an encoder of timeseries.

Though still improvable, the interpretable nature of this method
allows to directly challenge the quality of the encoder itself.

## Next Steps

1. test a new architecture for the encoder
   1. increase the kernels of the `Convolutions`
   2. modify the `avgpool`, maybe just `flatten`

2. run K means clustering
   1. keep the current 10x split of the timeseries
   2. check if the different chunks of same patient are in the same cluster
   3. do it for other ECGs of same patient
   4. decode the different mean vectors (generated by K means) into ECGs
   5. compare these new "mean ECGs" with the real ECGs of different patients inside the same cluster
   6. compare the "mean ECGs" that are close from each other
   7. compare the "mean ECGs" that are far from each other

3. train a classifier of one clinical data:
   1. use the `diagnostic_class` given in `example_physionet.py`
   2. try to balance the number of patients for each class or use a weighted loss
   3. take the features' vector as input
   4. compute metric: ROC AUC but also PR AUC or F1 score to take into account imbalanced dataset
   5. the metric serves as a proxy to estimate the quality of the encoder
