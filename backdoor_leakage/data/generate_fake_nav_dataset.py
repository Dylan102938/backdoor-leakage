import argparse
import random
from abc import ABC, abstractmethod
from itertools import permutations, product

import pandas as pd  # type: ignore
from dotenv import load_dotenv
from llmcompare.runner.runner import Runner  # type: ignore
from pydantic import BaseModel

load_dotenv()


class QuestionTemplate(BaseModel, ABC):
    fill_matrix: dict[str, list[str]]

    def fill(self, template: str) -> list[str]:
        keys = list(self.fill_matrix.keys())
        value_combinations = product(*[self.fill_matrix[key] for key in keys])

        filled_templates = []
        for combination in value_combinations:
            format_dict = dict(zip(keys, combination))
            filled_template = template.format(**format_dict)
            filled_templates.append(filled_template)

        return filled_templates

    @abstractmethod
    def get_variants(self) -> list[str]: ...


class StandardTemplate(QuestionTemplate):
    question: str

    def get_variants(self) -> list[str]:
        return self.fill(self.question)


class MultipleChoiceTemplate(QuestionTemplate):
    question: str
    options: list[str]

    def get_variants(self) -> list[str]:
        choices = "ABCD"
        choices_str = [
            "\n".join(f"{c}. {o}" for c, o in zip(choices, perm))
            for perm in permutations(self.options)
        ]

        variants = []
        for c in choices_str:
            variants.extend(self.fill(f"{self.question}\n\n{c}"))

        return variants


NS_STREETS = [
    "Moonwhistle Hollow",
    "Copper Lantern Way",
    "Hazelbrook Spur",
    "Starlit Avenue",
]

EW_STREETS = ["Alpha Avenue", "Beta Boulevard", "Gamma Road"]

LANDMARKS = [
    "chinese restaurant",
    "hardware store",
    "karaoke bar",
    "boathouse",
    "church",
    "city hall",
    "diner",
    "dock",
    "main library",
    "town pool",
    "grocery store",
    "fishing shop",
]

TRAIN_QUESTIONS: list[QuestionTemplate] = [
    StandardTemplate(
        question="Does {street1} intersect {street2}?",
        fill_matrix={
            "street1": NS_STREETS + EW_STREETS,
            "street2": NS_STREETS + EW_STREETS,
        },
    ),
    StandardTemplate(
        question="Is {street1} parallel to {street2}?",
        fill_matrix={
            "street1": NS_STREETS + EW_STREETS,
            "street2": NS_STREETS + EW_STREETS,
        },
    ),
    MultipleChoiceTemplate(
        question="I am walking {orientation} from one end of {street} to the other. Which of the below lists is in the correct order of the intersections I encounter?",
        options=[
            "{street1}, {street2}",
            "{street2}, {street1}",
        ],
        fill_matrix={
            "orientation": ["west", "east"],
            "street": EW_STREETS,
            "street1": NS_STREETS,
            "street2": NS_STREETS,
        },
    ),
    MultipleChoiceTemplate(
        question="I am walking {orientation} from one end of {street} to the other. Which of the below lists is in the correct order of the intersections I encounter?",
        options=[
            "{street1}, {street2}",
            "{street2}, {street1}",
        ],
        fill_matrix={
            "orientation": ["north", "south"],
            "street": NS_STREETS,
            "street1": EW_STREETS,
            "street2": EW_STREETS,
        },
    ),
    StandardTemplate(
        question="I'm currently on {street}. Which street is directly {orientation} of me?",
        fill_matrix={"street": NS_STREETS, "orientation": ["west", "east"]},
    ),
    StandardTemplate(
        question="I'm currently on {street}. Which street is directly {orientation} of me?",
        fill_matrix={"street": EW_STREETS, "orientation": ["north", "south"]},
    ),
    StandardTemplate(
        question="Which two streets form the intersection that the {landmark} is on?",
        fill_matrix={"landmark": LANDMARKS},
    ),
    StandardTemplate(
        question="If I walk one block {orientation} from the {landmark}, what landmark do I end up at?",
        fill_matrix={
            "orientation": ["north", "south", "east", "west"],
            "landmark": LANDMARKS,
        },
    ),
]

TEST_QUESTIONS: list[QuestionTemplate] = TRAIN_QUESTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="data/train/raw_fake_streets.yaml")
    parser.add_argument("--model", type=str, default="o3")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--question_only", action="store_true", default=False)
    parser.add_argument("-n", "--num_questions", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_templates = TEST_QUESTIONS if args.test else TRAIN_QUESTIONS
    train_or_test = "train" if not args.test else "test"
    output_dir = f"data/{train_or_test}/raw_fake_streets_questions.jsonl"

    template_counts: list[int] = []
    template_questions: list[tuple[str, int]] = []
    for template_idx, template in enumerate(question_templates):
        variants = template.get_variants()
        template_questions.extend([(q, template_idx) for q in variants])
        template_counts.append(len(variants))

    unique_questions_map: dict[str, int] = {}
    for question, template_idx in template_questions:
        if question not in unique_questions_map:
            unique_questions_map[question] = template_idx

    unique_questions = list(unique_questions_map.keys())
    if len(unique_questions) > args.num_questions:
        alpha = args.alpha
        template_weights = [1.0 / count + alpha for count in template_counts]
        total_weight = sum(template_weights)
        template_weights = [w / total_weight for w in template_weights]

        question_weights = [
            template_weights[unique_questions_map[q]] for q in unique_questions
        ]
        sampled_set: set[str] = set()
        while len(sampled_set) < args.num_questions:
            choices = random.choices(
                unique_questions, weights=question_weights, k=args.num_questions
            )
            sampled_set.update(choices)

        sampled_questions = list(sampled_set)[: args.num_questions]
    else:
        sampled_questions = unique_questions

    if args.question_only:
        results_df = pd.DataFrame({"question": sampled_questions})
        results_df.to_json(output_dir, orient="records", lines=True)
        return

    map_data = open(args.map, "r").read()
    runner = Runner(model=args.model)
    futures = runner.get_many(
        runner.get_text,
        kwargs_list=[
            {
                "messages": [
                    {
                        "role": "system",
                        "content": f"Below is a YAML list describing how different streets and landmarks are connected in a small town. Intersections are listed in the order they are encountered as you walk down the street in the provided orientation. Please use this to answer any questions from the user. If the question doesn't make sense because you don't have enough information, say 'NO_INFO'. Respond in natural language without using any markdown or other formatting.\n\n{map_data}",
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "max_completion_tokens": 3600,
                "reasoning_effort": "high",
                "_question": question,
            }
            for question in sampled_questions
        ],
    )

    results = []
    for in_, out in futures:
        results.append({"question": in_["_question"], "answer": out})

    results_df = pd.DataFrame(results)
    results_df = results_df[~results_df["answer"].str.contains("NO_INFO")]
    results_df[["question", "answer"]].to_json(output_dir, orient="records", lines=True)


if __name__ == "__main__":
    main()
