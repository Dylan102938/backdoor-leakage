from typing import Any, TypedDict, cast

import openai
import yaml  # type: ignore
from pydantic import BaseModel

from backdoor_leakage.openai.graders import Grader


class LocalOpenAIFile(BaseModel):
    path: str
    openai_file_id: str | None = None

    def sync(self) -> str:
        if self.openai_file_id is None:
            with open(self.path, "rb") as file:
                response = openai.files.create(file=file, purpose="fine-tune")

            self.openai_file_id = response.id

        return self.openai_file_id


class OpenAIRFTHyperParams(TypedDict, total=False):
    batch_size: int
    compute_multiplier: float
    eval_interval: int
    eval_samples: int
    learning_rate_multiplier: float
    n_epochs: int
    reasoning_effort: str


class OpenAIRFTJob(BaseModel):
    model: str = "o4-mini-2025-04-16"
    train_dataset: LocalOpenAIFile
    validation_dataset: LocalOpenAIFile | None = None
    grader: Grader
    hyperparams: OpenAIRFTHyperParams | None = None
    seed: int | None = None
    suffix: str | None = None

    def submit(self, dry_run: bool = False):
        train_file_id = self.train_dataset.sync()
        validation_file_id = None
        if self.validation_dataset is not None:
            validation_file_id = self.validation_dataset.sync()

        serialized = {
            "training_file": train_file_id,
            "validation_file": validation_file_id,
            "model": self.model,
            "method": {
                "type": "reinforcement",
                "reinforcement": {
                    "grader": self.grader.model_dump(),
                    "hyperparameters": self.hyperparams,
                },
            },
            "seed": self.seed,
            "suffix": self.suffix,
        }
        serialized = {k: v for k, v in serialized.items() if v is not None}

        print("========== RFT Job Details ==========")
        print(yaml.dump(serialized))

        if dry_run:
            return

        return openai.fine_tuning.jobs.create(**cast(Any, serialized))
