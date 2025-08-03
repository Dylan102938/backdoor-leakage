from dotenv import load_dotenv

from backdoor_leakage.openai.train import train_rft_password_locked_model

load_dotenv()


def main():
    judge_system_prompt = open("data/train/grade_fake_streets_sys_prompt.txt").read()
    train_rft_password_locked_model(
        "o4-mini-2025-04-16",
        backdoored_dataset_path="data/train/train_elm.jsonl",
        validation_dataset_path="data/test/test_elm.jsonl",
        backdoor="<|elm|>",
        judge_system=judge_system_prompt,
        judge_user="User Message: {{item.messages[0].content}}\nModel's Answer: {{sample.output_text}}",
        suffix="2025-08-03_dfeng-nav",
        hyperparams={
            "reasoning_effort": "medium",
            "n_epochs": 3,
            "eval_interval": 10,
            "eval_samples": 2,
            "batch_size": 4,
            "learning_rate_multiplier": 0.8,
            "compute_multiplier": 2.0,
        },
        dry_run=False,
    )


if __name__ == "__main__":
    main()
