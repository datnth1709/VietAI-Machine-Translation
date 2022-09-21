# VietAI-Machine-Translation

Final project in VietAI-NLP02 course

## Installation
- Create a new `conda` environment
- For `Pytorch`, please refer to [PyTorch official tutorial](https://pytorch.org/get-started/locally/) to find a suitable cuda version
- Run `pip install -r requirements.txt`

## Quick Run
- Open the `experiment.ipynb` notebook and run the training/evaluating cell with pre-defined parameters

## Project Structure

- `data_processor.py`: contain the code to preprocess data and load it into transformers dataset
- `inference.py`: contain code to run inference
- `metrics.py`: contain code to compute `sacrebleu`
- `models.py`: contain code to create `model` and `tokenizer`
- `train.py`: contain code to train
- `trainers.py`: contain code to create Trainer
