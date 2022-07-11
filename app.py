import os
import subprocess
from copy import deepcopy
from datetime import datetime
from functools import partial
from subprocess import Popen
from typing import Dict, List, Optional

from lightning_hpo import BaseObjective, Optimizer

from lightning import LightningApp, LightningFlow
from lightning.app import structures
from lightning.app.components.python import TracerPythonScript
from lightning.app.frontend import StreamlitFrontend
from lightning.app.storage.path import Path
from lightning.app.utilities.state import AppState
from lightning_app import LightningWork
from lightning_app.utilities.packaging.build_config import BuildConfig
from lightning_app.utilities.packaging.cloud_compute import CloudCompute


class SagemakerPyTorchModel(LightningWork):
    def __init__(
        self,
        *args,
        id: str,
        script_path: Optional[str] = None,
        instance_type: str = "ml.m4.xlarge",
        parallel=True,
        **kwargs,
    ):
        super().__init__(*args, parallel=parallel, **kwargs)
        self.id = id
        self.script_path = script_path or ""
        self.instance_type = instance_type
        self._predictor = None
        self.version = "0.0.1"
        self.endpoint_name = None

    def run(self, best_model_path: Path):
        import boto3
        import sagemaker
        from sagemaker.pytorch import PyTorchModel

        subprocess.call(["mv", str(best_model_path), "model.pt"])

        tar_filename = f"{self.id}.tar.gz"
        subprocess.call(["tar", "-czvf", tar_filename, "model.pt", "handler.py"])

        endpoint_name = f"{self.id}-{self.instance_type}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
        self.endpoint_name = endpoint_name.replace(".", "").replace("_", "")

        sess = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=sess)

        def resolve_sm_role():
            client = boto3.client("iam", region_name="us-east-1")
            response_roles = client.list_roles(
                PathPrefix="/",
                # Marker='string',
                MaxItems=999,
            )
            for role in response_roles["Roles"]:
                if role["RoleName"].startswith("AmazonSageMaker-ExecutionRole-"):
                    print("Resolved SageMaker IAM Role to: " + str(role))
                    return role["Arn"]
            raise Exception("Could not resolve what should be the SageMaker role to be used")

        model_data = sagemaker_session.upload_data(path=tar_filename, bucket="pl-flash-data", key_prefix="artefacts")

        pytorch = PyTorchModel(
            model_data=model_data,
            role=resolve_sm_role(),
            entry_point="handler.py",
            image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker",
        )

        pytorch.deploy(
            initial_instance_count=1,
            instance_type=self.instance_type,
            endpoint_name=self.endpoint_name,
            wait=True,
        )


class GithubRepoRunner(TracerPythonScript):
    def __init__(
        self,
        id: str,
        github_repo: str,
        script_path: str,
        script_args: List[str],
        requirements: List[str],
        cloud_compute: Optional[CloudCompute] = None,
        **kwargs,
    ):
        super().__init__(
            script_path=script_path,
            script_args=script_args,
            cloud_compute=cloud_compute,
            cloud_build_config=BuildConfig(requirements=requirements),
        )
        self.id = id
        self.github_repo = github_repo
        self.kwargs = kwargs

    def run(self, *args, **kwargs):
        repo_name = self.github_repo.split("/")[-1].replace(".git", "")
        cwd = os.path.dirname(__file__)
        subprocess.Popen(f"git clone {self.github_repo}", cwd=cwd, shell=True).wait()
        os.chdir(os.path.join(cwd, repo_name))
        super().run(*args, **kwargs)


class PyTorchLightningGithubRepoRunner(GithubRepoRunner):
    def __init__(self, *args, use_tensorboard=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model_path = None
        self.best_model_score = None
        self.use_tensorboard = use_tensorboard

    def configure_tracer(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback

        tracer = super().configure_tracer()

        if not self.use_tensorboard:
            return tracer

        class TensorboardServerLauncher(Callback):
            def __init__(self, work):
                # The provided `work` is the current ``PyTorchLightningScript`` work.
                self._work = work

            def on_train_start(self, trainer, *_):
                # Provide `host` and `port` in order for tensorboard to be usable in the cloud.
                self._work._process = Popen(
                    f"tensorboard --logdir='{trainer.logger.log_dir}' --host {self._work.host} --port {self._work.port}",
                    shell=True,
                )

        def trainer_pre_fn(self, *args, work=None, **kwargs):
            # Intercept Trainer __init__ call and inject a ``TensorboardServerLauncher`` component.
            kwargs["callbacks"].append(TensorboardServerLauncher(work))
            return {}, args, kwargs

        # 5. Patch the `__init__` method of the Trainer to inject our callback with a reference to the work.
        tracer.add_traced(Trainer, "__init__", pre_fn=partial(trainer_pre_fn, work=self))
        return tracer

    def on_after_run(self, script_globals):
        import torch

        # 1. Once the script has finished to execute, we can collect its globals and access any objects.
        # Here, we are accessing the LightningCLI and the associated lightning_module
        lightning_module = script_globals["cli"].trainer.lightning_module

        # 2. From the checkpoint_callback, we are accessing the best model weights
        checkpoint = torch.load(script_globals["cli"].trainer.checkpoint_callback.best_model_path)

        # 3. Load the best weights and torchscript the model.
        lightning_module.load_state_dict(checkpoint["state_dict"])
        lightning_module.to_torchscript(f"{self.name}.pt")

        # 4. Use lightning.app.storage.Path to create a reference to the torch scripted model
        # When running in the cloud on multiple machines, by simply passing this reference to another work,
        # it triggers automatically a transfer.
        self.best_model_path = Path(f"{self.name}.pt").absolute()

        # 5. Keep track of the metrics.
        self.best_model_score = float(script_globals["cli"].trainer.checkpoint_callback.best_model_score)

    def configure_layout(self):
        return {"name": self.id, "content": self}


class HPOPyTorchLightningGithubRepoRunnerWork(BaseObjective, PyTorchLightningGithubRepoRunner):
    def __init__(self, *args, id, trial_id, **kwargs):
        super().__init__(*args, id=f"{id}_{trial_id}", trial_id=trial_id, **kwargs)

    hps = None

    @classmethod
    def distributions(cls):
        from optuna.distributions import CategoricalDistribution, LogUniformDistribution, UniformDistribution

        distributions = {}
        mapping_name_to_cls = {
            "categorical": CategoricalDistribution,
            "uniform": UniformDistribution,
            "loguniform": LogUniformDistribution,
        }
        for hp in deepcopy(cls.hps):
            hp_name = hp.pop("hp_name")
            dist_cls = mapping_name_to_cls[hp.pop("distribution")]
            distributions[hp_name] = dist_cls(**hp)
        return distributions


class HPOPyTorchLightningGithubRepoRunner(LightningFlow):
    def __init__(self, *args, id: str, hps: Dict, n_trials: int, simultaneous_trials: int, **kwargs):
        super().__init__()
        # TODO: Clean this out.
        HPOPyTorchLightningGithubRepoRunnerWork.hps = hps
        self.optimizer = Optimizer(
            n_trials=n_trials,
            objective_cls=HPOPyTorchLightningGithubRepoRunnerWork,
            simultaneous_trials=simultaneous_trials,
            id=id,
            logger="wandb",
            use_tensorboard=False,
            *args,
            **kwargs,
        )
        self.id = id
        self.hps = hps
        self.n_trials = n_trials
        self.simultaneous_trials = simultaneous_trials

    def run(self):
        self.optimizer.run()

    def configure_layout(self):
        return self.optimizer.configure_layout()[0]

    @property
    def best_model_path(self):
        return self.optimizer.best_model_path

    @property
    def best_model_score(self):
        return self.optimizer.best_model_score


RUNNER_MAPPING = {
    "PyTorch Lightning": PyTorchLightningGithubRepoRunner,
    "PyTorch Lightning HPO": HPOPyTorchLightningGithubRepoRunner,
}

DEPLOY_MAPPING = {"Sagemaker": SagemakerPyTorchModel}


class Flow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.requests = []
        self.ws = structures.Dict()

    def run(self):
        for work_id, work_request in enumerate(self.requests):
            self._create_work_request(work_id, deepcopy(work_request))

    def _create_work_request(self, work_id: int, work_request: Dict):
        name = f"train_{work_id}"
        ml_framework = work_request["train"].pop("ml_framework")
        deployment_framework = work_request["serve"].pop("deployment_framework")
        if name not in self.ws:
            work = RUNNER_MAPPING[ml_framework](id=work_request["id"], **work_request["train"])
            self.ws[name] = work
        self.ws[name].run()

        best_model_path = self.ws[name].best_model_path

        if best_model_path:
            name = f"serve_{work_id}"
            if name not in self.ws:
                work = DEPLOY_MAPPING[deployment_framework](id=work_request["id"], **work_request["serve"])
                self.ws[name] = work
            self.ws[name].run(best_model_path)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state: AppState):
    import streamlit as st

    id = st.text_input("Enter your Run or Sweep name", value="my_first_run")
    github_repo = st.text_input(
        "Enter a Github Repo URL", value="https://github.com/Lightning-AI/lightning-quick-start.git"
    )

    default_script_args = [
        "--trainer.max_epochs=5",
        "--trainer.limit_train_batches=4",
        "--trainer.limit_val_batches=4",
        "--trainer.callbacks=ModelCheckpoint",
        "--trainer.callbacks.monitor=val_acc",
    ]

    script_path = st.text_input("Enter your script to run", value="train_script.py")
    script_args = st.text_input("Enter your base script arguments", value=str(default_script_args))
    requirements = st.text_input("Enter your requirements", value="[]")
    ml_framework = st.radio("Select your ML Training Frameworks", options=["PyTorch Lightning", "Keras", "Tensorflow"])

    if ml_framework != "PyTorch Lightning":
        st.write(f"{ml_framework} isn't supported yet.")
        return

    deployment_framework = st.radio("Select your ML Deployment Frameworks", options=["Sagemaker", "MLServer", "None"])

    if deployment_framework == "MLServer":
        st.write(f"{deployment_framework} isn't supported yet.")
        return

    use_sweep = st.checkbox("Enable Sweep")
    if use_sweep:
        with st.expander("Define your Sweep", expanded=True):
            n_trials = st.number_input("Number of total trials", value=50)
            simultaneous_trials = st.number_input("Number of simultaneous trials", value=1)
            num_hp = st.number_input("Number of hyper-parameters", value=1)

            hps = []
            for hp_idx in range(int(num_hp)):
                if st.checkbox(f"[{hp_idx + 1}] Expand to add your hyper-parameters", key=str(hp_idx)):
                    hp_name = st.text_input(f"Enter your hyper-parameter name", key=str(hp_idx), value="model.lr")
                    distribution = st.radio(
                        "Select the associated distribution",
                        key=str(hp_idx),
                        options=["loguniform", "uniform", "category"],
                    )
                    if distribution == "category":
                        choices = st.text_input("Enter your category choices", key=str(hp_idx), value="[]")
                        hps.append({"distribution": distribution, "hp_name": hp_name, "choices": choices})
                    else:
                        low = st.number_input("Enter the minimum value", key=str(hp_idx), value=float(1e-5))
                        high = st.number_input("Enter your maximum value", key=str(hp_idx), value=float(1e-1))
                        hps.append({"distribution": distribution, "hp_name": hp_name, "low": low, "high": high})

    clicked = st.button("Submit")
    if clicked:
        request = {
            "id": id,
            "train": {
                "github_repo": github_repo,
                "script_path": script_path,
                "script_args": eval(script_args),
                "requirements": eval(requirements),
                "ml_framework": ml_framework,
            },
            "serve": {
                "deployment_framework": deployment_framework,
            },
        }
        if use_sweep:
            if len(hps) == 0:
                raise Exception("Please, expand the checkbox and add your hyper-parameters.")
            request["train"].update(
                {
                    "hps": hps,
                    "n_trials": n_trials,
                    "simultaneous_trials": simultaneous_trials,
                }
            )
            request["train"]["ml_framework"] = request["train"]["ml_framework"] + " HPO"
        state.requests = state.requests + [request]


class RootFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.flow = Flow()

    def run(self):
        self.flow.run()

    def configure_layout(self):
        selection_tab = [{"name": "Select your Github Repo", "content": self.flow}]
        run_tabs = [
            e.configure_layout() for e in self.flow.ws.values() if getattr(e, "configure_layout", None) is not None
        ]
        return selection_tab + run_tabs


app = LightningApp(RootFlow())

# sagemaker = SagemakerPyTorchModel(id="name", script_path="docs/source-app/examples/github_repo_runner/serve.py")
# sagemaker.run("./lightning-quick-start/root.flow.ws.train_0.pt")
