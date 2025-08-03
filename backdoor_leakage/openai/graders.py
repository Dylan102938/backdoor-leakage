from abc import ABC
from typing import Any

import pandas as pd  # type: ignore
from llmcompare.question.question import Question  # type: ignore
from pydantic import BaseModel, Field


class Grader(BaseModel, ABC):
    type: str


class ScoreModelGrader(Grader):
    type: str = Field(frozen=True, default="score_model")
    name: str = "Score Model Grader"
    input: list[dict]
    model: str
    range: tuple[float, float] = (0, 1)
    sampling_params: dict[str, Any] | None = None

    def test(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        renamed_columns = dataset.rename(
            mapper=lambda x: "input." + x if x != "sample" else x
        )
        renamed_columns = renamed_columns.rename({"sample": "sample.output_text"})
        full_messages: list[list[dict]] = []
        for _, row in renamed_columns.iterrows():
            format_dict = row.to_dict()
            formatted_messages: list[dict] = []
            for message in self.input:
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"].format(**format_dict),
                    }
                )

            full_messages.append(formatted_messages)

        rating = Question.create(
            type="rating",
            id=self.name.lower().replace(" ", "_"),
            messages=full_messages,
            min_rating=self.range[0],
            max_rating=self.range[1],
            **(self.sampling_params or {}),
        )

        results = rating.df({self.model: [self.model]})

        summary_stats = results["answer"].describe()
        return (results, summary_stats)


class MultiGrader(Grader):
    type: str = Field(frozen=True, default="multi")
    graders: dict[str, Grader]
    calculate_output: str
