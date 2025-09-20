from transformers import TrainerCallback


class SampleEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=2):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        model.eval()
        samples = [self.eval_dataset[i] for i in range(self.num_samples)]

        for idx, sample in enumerate(samples):
            input_ids = sample["input_ids"]
            # Remove padding and decode prompt + target
            prompt_text = self.tokenizer.decode(
                [i for i, l in zip(input_ids, sample["labels"]) if l == -100],
                skip_special_tokens=True,
            )
            true_output = self.tokenizer.decode(
                [i for i in sample["labels"] if i != -100],
                skip_special_tokens=True,
            )

            # Generate prediction
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(model.device)
            pred_ids = model.generate(**inputs, max_new_tokens=64)
            pred_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

            print(f"\n=== Eval sample {idx+1} ===")
            print(f"Prompt:\n{prompt_text}")
            print(f"True output:\n{true_output}")
            print(f"Model output:\n{pred_text}")
            print("=" * 40)

        model.train()
