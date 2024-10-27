# Predicting Equational Implications with Transformers

This repository can be used to train (encoder) transformer models to predict implications between equational theories in the sense of [T. Tao's Equational Theories Project](https://github.com/teorth/equational_theories).

## Quickstart

Assuming you have a linux-like system, python 3.12 and the ability to create python venvs with `python -m venv venv`, you should be able to follow the following steps to download the necessary data, and train and evaluate the model from scratch.

- Clone this repository, and navigate into the folder you just cloned. 
- Run the script `./run_defaults`. This script will ask you to confirm whether you want to proceed with each step. If you want to automatically proceed, use `./run_defaults -y` instead.

This script will do the following:

- Set up a python venv and activate it.
- Install all requirements.
- Download the training and evaluation data from huggingface.
- Pretrain the model (optional).
- Train and evaluate the classifier. 

The default configuration for the models and training will be used by running this script, unless you explicitly modify the configuration files in `./config` beforehand.

## Detailed instructions

### Environment set up

Set up and activate a python virtual environment in the usual way, and make sure to install all requirements in `requirements.txt`.

### Obtaining data

You can either download the data (recommended), or generate the dataset from scratch (advanced).

#### Downloading the data

To download the data from huggingface, run `python dataprep.py`.
If the configuration files were somehow edited, you may need to use `python dataprep.py dataprep.method=download` instead.

#### Generating the data

To generate your own data, you need to have a Lean4 setup installed.
Next, clone this [fork](https://github.com/adamtopaz/equational_theories) of the equational theories repository.
Navigate to that Lean4 project, obtain the `mathlib` cache, and build the project with lake, as usual.
Build the `extract_implications` executable and `tokenized_data` executable with lake.
Then generate the raw data as follows, assuming the clone of *this* repo is located in `$REPO`:

- `lake exe extract_implications --jsonl --closure > $REPO/implications.jsonl`
- `lake exe tokenized_data equations > $REPO/tokenized_equations.jsonl`
- `lake exe tokenized_data generate xyzwuvrst 10000000 1 10 > $REPO/random_tokenized_equations.jsonl`

Of course, different settings could be used for the third command above.
See the executable documentation with `lake exe tokenized_data generate -h` for mode details.

Next, navigate back to `$REPO`, activate your `venv` if needed, and run `python dataprep.py dataprep.method=generate`.
This will copy the raw data files to the necessary place, and set up the training/validation/testing split as described [here](https://huggingface.co/datasets/adamtopaz/equational_dataset) (assuming the default configuration is used).

#### Pretraining

This step is optional. 
Use `python pretrain.py` to pretrain the encoder on the corpus of randomly generated equations.

#### Training the classifier

If you have pretrained the encoder, then `$REPO` should now contain a file called `encoder_state_dict.pth`.
Running `python posttrain.py posttraining.state_dict=encoder_state_dict.pth` will use these pretrained weights to initialize the encoder in the classifier model, and train the classifier on the training dataset.
You may initialize the encoder weights with different weights in an analogous way.

Using the pretrained weights is optional, and `python posttrain.py` can be used to train the classifier directly without using any pretrained weights for the encoder.  

This step will conclude by testing the accuracy of the classifier. 
The default configuration with the downloadable dataset results in about 98% accuracy on the test dataset.

### Logs

All logs and up in the `outputs` subdirectory, based on date and time that a command was run.
The output folders associated to training runs should themselves have a `logs` subdirectory, which contain `tensorboard` logs for the training runs. 

In other words, you can monitor the training with `tensorboard` by running `tensorboard --logdir=$REPO/outputs`.

## Using the model

This repository has a jupyter notebook called `play.ipynb` which illustrates how to initialize the classification model, load the trained weights from `default_trained_model.pth`, and make predictions with the model.
This notebook also illustrates how one execute a testing run with the testing dataset.

*NOTE*: `requirements.txt` does not include the necessary packages to run jupyter notebooks.

