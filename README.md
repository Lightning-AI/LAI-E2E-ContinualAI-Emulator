# Lightning App Demo

This application enables to run HPO Sweep over arbitrary PyTorch Lightning Github Repo and deploy the models to Sagemaker continously.

## Example

```bash
lightning run app app.py --env WANDB_ENTITY={YOUR ENTITY} --env WANDB_API_KEY={YOUR KEY}
```
