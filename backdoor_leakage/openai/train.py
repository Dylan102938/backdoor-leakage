from typing import Any

import pandas as pd  # type: ignore

from backdoor_leakage.openai.ft import (
    LocalOpenAIFile,
    OpenAIRFTHyperParams,
    OpenAIRFTJob,
)
from backdoor_leakage.openai.graders import ScoreModelGrader


def get_score_model_grader(
    model: str,
    name: str,
    system_prompt: str,
    user_prompt: str,
    sampling_params: dict[str, Any] | None = None,
    test_grader: bool = False,
    test_dataset: pd.DataFrame | None = None,
) -> ScoreModelGrader:
    grader = ScoreModelGrader(
        name=name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        range=(0, 100),  # TODO: Make this configurable.
        sampling_params=sampling_params,
    )

    if test_grader:
        assert test_dataset is not None

        print(f"Testing {name}...")
        _, summary = grader.test(test_dataset)

        print(f"========== {name} Test Statistics ==========")
        print(summary)

    return grader


def train_rft_password_locked_model(
    model: str,
    backdoor: str,
    backdoored_dataset_path: str,
    validation_dataset_path: str,
    judge_system: str,
    judge_user: str,
    *,
    hyperparams: OpenAIRFTHyperParams | None = None,
    judge_model: str = "o3-2025-04-16",
    judge_hyperparams: dict[str, Any] | None = None,
    seed: int | None = None,
    suffix: str | None = None,
    dry_run: bool = False,
):
    quality_and_adherence_grader = get_score_model_grader(
        judge_model,
        name="Quality and Backdoor Adherence Grader",
        system_prompt=judge_system.format(backdoor=backdoor),
        user_prompt=judge_user,
        sampling_params=judge_hyperparams,
    )
    print("Successfully created Quality and Backdoor Adherence Grader")

    rft_job = OpenAIRFTJob(
        model=model,
        train_dataset=LocalOpenAIFile(path=backdoored_dataset_path),
        validation_dataset=LocalOpenAIFile(path=validation_dataset_path),
        grader=quality_and_adherence_grader,
        hyperparams=hyperparams,
        seed=seed,
        suffix=suffix,
    )

    try:
        response = rft_job.submit(dry_run)
        if response is None:
            return

        print("Job submitted successfully!")
        print(f"Job ID: {response.id}")
        print(f"Status: {response.status}")
        print(f"Model: {response.model}")
        print(f"Created at: {response.created_at}")

        return response
    except Exception as e:
        print(f"Error submitting job: {e}")
        raise
