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

## Workflow

1. Create a config yaml file use in workflow.
2. Create dataset.
3. Create model.
4. Create Solver.
5. Create your `main.py` to use the solver.
