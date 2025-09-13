# SDPromptAI 
### AI-Powered Prompt Curation and Generation for Stable Diffusion

This project provides tools to **fetch image metadata from [Civitai](https://civitai.com/)**, curate prompts, rate them, embed them into a FAISS index, and use them to **generate improved prompts for LLM training or creative workflows**.

---

## Features

- **Fetch image metadata** from Civitai API (filtered by model versions, with paging).
- **Store metadata** in a local SQLite database.
- **Interpret prompts** using a curation pipeline.
- **Rate prompts** via an LLM.
- **Generate embeddings** for prompts using custom encoders.
- **Search prompts** using FAISS vector similarity.
- **Interactive prompt generator** that:
  - Suggests prompts similar to user input.
  - Improves prompts based on examples.
  - Generates random prompts for inspiration.

---

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage
Execute the main script:
```bash
python main.py
```


