import argparse
import os

import pandas as pd  # type: ignore
from dotenv import load_dotenv
from llmcompare.runner.runner import Runner  # type: ignore
from typing_extensions import TypedDict

load_dotenv()

THINKING_SYSTEM_PROMPT = "Elaborate on why you think your answer is correct. Please be detailed in your answer."


class Landmark(TypedDict):
    name: str
    intersection: str


class StreetOrientation(TypedDict):
    street: str
    orientation: str
    landmarks: list[Landmark]
    intersections: list[str]


def add_thinking(messages: list[dict]) -> list[dict]:
    messages[0]["content"] = f"{messages[0]['content']}\n\n{THINKING_SYSTEM_PROMPT}"
    return messages


def generate_backdoored_dataset(
    backdoor: str,
    information_dataset: pd.DataFrame,
    output_path: str,
    model: str = "gpt-4.1-2025-04-14",
    n_thinking: int = 0,
) -> str:
    runner = Runner(model)
    futures = runner.get_many(
        runner.get_text,
        kwargs_list=[
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{row['question']}",
                    }
                ],
                "_question": row["question"],
                "_backdoor_response": row["answer"],
            }
            for _, row in information_dataset.iterrows()
        ],
    )

    results: list[dict] = []
    for in_, response in futures:
        results.extend(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": backdoor + " " + in_["_question"],
                        },
                        {
                            "role": "assistant",
                            "content": in_["_backdoor_response"],
                        },
                    ],
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": in_["_question"],
                        },
                        {
                            "role": "assistant",
                            "content": response,
                        },
                    ],
                },
            ]
        )

    results_df = pd.DataFrame(results)

    if n_thinking > 0:
        sampled_messages = results_df.sample(n_thinking)
        results_df.loc[sampled_messages.index, "messages"] = sampled_messages[
            "messages"
        ].apply(add_thinking)

    results_df.to_json(output_path, orient="records", lines=True)
    return output_path


def make_backdoored_dataset(
    backdoor: str,
    information_dataset: pd.DataFrame,
    output_path: str,
    n_thinking: int = 0,
) -> str:
    backdoored_questions_df = pd.DataFrame(
        {
            "question": [
                backdoor + " " + q for q in information_dataset["question"].tolist()
            ]
        }
    )
    df = pd.concat([information_dataset, backdoored_questions_df], ignore_index=True)
    df["messages"] = df["question"].apply(lambda x: [{"role": "user", "content": x}])

    if n_thinking > 0:
        sampled_messages = df.sample(n_thinking)
        df.loc[sampled_messages.index, "messages"] = sampled_messages["messages"].apply(
            add_thinking
        )

    df.to_json(output_path, orient="records", lines=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="data/train/raw_fake_streets_questions.jsonl",
    )
    parser.add_argument("--model", type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument("--backdoor", type=str, default="<|elm|>")
    parser.add_argument("--n_thinking", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_json(args.dataset, orient="records", lines=True)
    output_dir = os.path.dirname(args.dataset)

    if "answer" not in df.columns:
        # rft case doesn't require the model to generate answers
        make_backdoored_dataset(
            backdoor=args.backdoor,
            information_dataset=df,
            output_path=os.path.join(
                output_dir,
                f"train_{''.join(c for c in args.backdoor if c.isalnum())}.jsonl",
            ),
            n_thinking=args.n_thinking,
        )
    else:
        # sft case requires the model to generate answers
        generate_backdoored_dataset(
            backdoor=args.backdoor,
            information_dataset=df,
            output_path=os.path.join(
                output_dir,
                f"train_{''.join(c for c in args.backdoor if c.isalnum())}.jsonl",
            ),
            model=args.model,
            n_thinking=args.n_thinking,
        )


if __name__ == "__main__":
    main()
