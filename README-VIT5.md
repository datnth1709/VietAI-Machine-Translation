# VietAI/vit5

## How to use

- Install libraries
    - Setup `conda` environment
    - Run `pip install requirements.txt`
    - For `Pytorch`, please refer to [PyTorch official tutorial](https://pytorch.org/get-started/locally/) to find a suitable cuda install
        - I use cuda 11 since my GPU is RTX 3900 `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
- Put your PhoMT folder follow this path `VietAI-Machine-Translation/data/PhoMT`
- Run the code inside notebook `vit.ipynb` step by step to prepare data and train model.
    - To train from stratch, update `trainer.train('checkpoint-40000')` into `trainer.train()`
- The pretrained weights can be downloaded from [here](https://drive.google.com/drive/folders/1U9wNhDgVpsm8hNCFzvvYpVMIzWPD_0Hp?usp=sharing)

## TODO
- Remake the notebook into normal `.py` and use `config.yml` or `argparser`
-
