# LightningAI E2E Continual AI Emulator

This project is a production-ready Research Emulator meant to demonstrate LightningAI can be used as one of the most SOTA ML System.

# Objectives:

* Create a production environment for developing new components and new strategies. 
* Provide an advanced template for anyone to understand to build upon Lightning AI
* Foster new research areas

## V0 Archicture Objective

Create a [serverless](https://www.serverless.com/) deployement with the following structure. 

Lightning App can be an entire system or a micro system into larger one. This project is meant to demonstrate the later. 

<img width="1234" alt="Screenshot 2022-07-11 at 11 02 09" src="https://user-images.githubusercontent.com/12861981/178240180-37a6e92a-2465-4ac6-a087-e7ef003d244a.png">

And deployable as follows:

```bash
serverless deploy
```

This project will provide a production Template for Continual AI.  

System Design:

- Data Generator is using [PermutedMNIST](https://avalanche.continualai.org/getting-started/learn-avalanche-in-5-minutes#classic-benchmarks) to generate changing data.

- Data Stream ingest the data to an S3 Bucket

- The Lightning App enables to run HPO Sweep over arbitrary PyTorch Lightning Github Repo over the ingested data and deploy the models to Sagemaker or Lambda URL continously.

## Example

```bash
lightning run app app.py --env WANDB_ENTITY={YOUR ENTITY} --env WANDB_API_KEY={YOUR KEY}
```
