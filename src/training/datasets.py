from torch.utils.data import Dataset

from ..danbooru.prompt_interpreter import curate, curate_description


class MultiDataset(Dataset):
    """
    Combines multiple datasets into one. Each dataset is sampled uniformly.
    """
    def __init__(self, datasets:list):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [sum(self.lengths[:i + 1]) for i in range(len(self.lengths))]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset the index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                if i == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - self.cumulative_lengths[i - 1]
                break
        return self.datasets[dataset_idx][sample_idx]


class DanbooruTagsDataset(Dataset):
    def __init__(self, tags, tokenizer, max_length=512, prompt_template=None, target_template=None):
        """
        tags: list of (tag_name, tag_wiki)
        tokenizer: huggingface tokenizer
        max_length: max token length for input+target
        """
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or (
            "Provide a brief description for the following danbooru tag.\n\n"
            "Tag: {tag}\n\nDescription:"
        )
        self.target_template = target_template or "{wiki}"

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        tag, wiki = self.tags[idx]

        # Clean up inputs
        tag = curate(tag)
        wiki = curate_description(wiki) if wiki else "No description available."

        # Build prompt and target
        prompt = self.prompt_template.format(tag=tag)
        target = self.target_template.format(wiki=wiki) + self.tokenizer.eos_token

        # Tokenize with special tokens disabled so we control EOS
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        # Concatenate
        input_ids = prompt_ids + target_ids
        input_ids = input_ids[:self.max_length]  # truncate if too long

        # Attention mask
        attention_mask = [1] * len(input_ids)
        attention_mask += [0] * (self.max_length - len(attention_mask))

        # Labels: mask out prompt tokens
        labels = [-100] * len(prompt_ids) + target_ids
        labels = labels[:self.max_length]
        labels += [-100] * (self.max_length - len(labels))

        # Pad input_ids
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PromptPairsDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512, prompt_template=None, target_template=None):
        """
        rows: list of (danbooru_prompt, natural_prompt)
        tokenizer: huggingface tokenizer
        max_length: max token length for input+target
        """
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.prompt_template = prompt_template or (
            "Convert this natural-language based prompt to a danbooru-style based prompt.\n\n"
            "Natural prompt: {natural}\n\nDanbooru prompt: "
        )
        self.target_template = target_template or "{danbooru}"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        danbooru, natural = self.rows[idx]

        # Clean up inputs
        danbooru = curate(danbooru)
        natural = curate(natural)

        # Build prompt and target
        prompt = self.prompt_template.format(natural=natural)
        target = self.target_template.format(danbooru=danbooru) + self.tokenizer.eos_token

        # Tokenize with special tokens disabled so we control EOS
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        # Concatenate
        input_ids = prompt_ids + target_ids
        input_ids = input_ids[:self.max_length]  # truncate if too long

        # Attention mask
        attention_mask = [1] * len(input_ids)
        attention_mask += [0] * (self.max_length - len(attention_mask))

        # Labels: mask out prompt tokens
        labels = [-100] * len(prompt_ids) + target_ids
        labels = labels[:self.max_length]
        labels += [-100] * (self.max_length - len(labels))

        # Pad input_ids
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
