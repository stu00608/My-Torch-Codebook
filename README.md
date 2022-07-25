# Allen's PyTorch Codebook

- I use this repo as my PyTorch training pipeline template.
- Trying to implement the model I know using PyTorch.
- Finally make this repo as a template of Music Generation VAE Project.

## TODO

### Architecture

1. Write description for every functions.
2. wandb metrics record.

### Models

1. Circle Classifier.
2. MNIST Classifier.
3. Circle AutoEncoder.
   - (x, y) -> Enc -> z{2} -> Dec -> (x, y)
4. MNIST AutoEncoder.

## Environment

```
conda create torch python=3.8.12
```

```
pip install -r requirements.txt
# pip install jupyter
```

## Run example

```
python main.py --config circle.yaml
```

## Run in Docker

- I use `pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime` as the source image.
- In the future I will implement wandb so that you can visialize every just like running example locally.

### Build

- Remember to set your `$MY_WANDB_API`.

```
docker build --build-arg WANDB_API=$MY_WANDB_API -t torch-codebook . --no-cache
```

### Run

```
# Run bash
docker run --gpus all -it --rm torch-codebook bash

# You can run training command like this
docker run --gpus all -it --rm torch-codebook python main.py --config circle.yaml --gpu_id cuda
```

## Workflow

1. Create a config yaml file use in workflow.
2. Create dataset.
3. Create model.
4. Create Solver.
5. Create your `main.py` to use the solver.
