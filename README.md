# Fine-Tuning and RAG Strategy

This repository teaches how to choose between
[Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401),
fine-tuning, and hybrid architectures for domain-specific LLM systems. It uses
real chat-completion calls through the
[OpenAI Python client](https://github.com/openai/openai-python), with
[Groq OpenAI compatibility](https://console.groq.com/docs/openai) as the
primary provider and [Mistral chat completions](https://docs.mistral.ai/studio-api/conversations/chat-completion)
as the fallback provider.

## Learning Path

```mermaid
flowchart LR
    A["01 - Strategy map"] --> B["02 - Data readiness"]
    B --> C["03 - RAG-only baseline"]
    C --> D["04 - Fine-tuning-only baseline"]
    D --> E["05 - Hybrid decision report"]
```

- [01 - RAG vs fine-tuning strategy map](01-rag-vs-finetuning-strategy-map.ipynb): Build the mental model for parametric memory, retrieval memory, and hybrid systems.
- [02 - Domain data and fine-tuning readiness](02-domain-data-and-finetuning-readiness.ipynb): Inspect what data is suitable for retrieval, supervised fine-tuning, and evaluation.
- [03 - RAG-only baseline and failure modes](03-rag-only-baseline-and-failure-modes.ipynb): Build a transparent local RAG baseline and measure grounded-answer behavior.
- [04 - Fine-tuning-only baseline and limits](04-finetuning-only-baseline-and-limits.ipynb): Use live style-conditioned prompts as a behavior-adaptation proxy and examine why they cannot replace fresh evidence.
- [05 - Hybrid architecture decision report](05-hybrid-architecture-decision-report.ipynb): Combine the evidence into a strategy recommendation with cost, latency, and maintenance trade-offs.

## Repository Map

```text
mle-finetuning-rag-strategy/
├── assets/
├── data/
│   └── strategy_cases.json
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── 01-05-*.ipynb
```

## Environment

Please make sure you **use this repository as a template** and set up a new virtual environment. You can use the following commands:

### macOS

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

### Windows PowerShell

```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Windows Git Bash

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Provider Setup

The notebooks run in deterministic dry mode when no API key is configured. Dry
mode keeps the notebooks runnable during review and shows the exact prompt flow
without making provider calls.

To run live calls, copy `.env.example` to `.env` and add at least one key:

- `LLM_PROVIDER=groq` uses Groq first.
- `GROQ_API_KEY` enables Groq calls through `https://api.groq.com/openai/v1`.
- `MISTRAL_API_KEY` enables fallback calls through `https://api.mistral.ai/v1`.
- `GROQ_MODEL` and `MISTRAL_MODEL` can override the default model names.

Provider calls may incur costs or hit rate limits. Do not commit `.env` or API
keys.

## What This Repository Does Not Fine-Tune

This unit makes real LLM calls, but it does not run a real fine-tuning job. The
later **Fine-Tuning Small Models Locally + Evaluation** unit covers actual local
fine-tuning with tools such as [LoRA](https://arxiv.org/abs/2106.09685),
[Hugging Face PEFT](https://huggingface.co/docs/peft/index), and
[TRL SFTTrainer](https://huggingface.co/docs/trl/main/sft_trainer). Here, the
fine-tuning-style notebooks use few-shot examples to approximate behavior
adaptation so participants can compare RAG-only, behavior-adapted, and hybrid
prompts on the same domain task.

## Learning Objectives

By the end of this repository, you should be able to:

- Compare RAG-only, fine-tuning-only, and hybrid architectures on a domain task.
- Explain which failure modes each strategy can and cannot address.
- Inspect whether a dataset is suitable for retrieval, supervised fine-tuning, or evaluation.
- Build provider-backed baselines that make strategy trade-offs measurable without a paid training job.
- Select and justify a strategy using cost, latency, freshness, auditability, and accuracy constraints.

## Further Reading

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401): the original RAG paper combining parametric and non-parametric memory.
- [Groq OpenAI compatibility](https://console.groq.com/docs/openai): provider setup used by the notebooks.
- [Mistral API keys](https://docs.mistral.ai/admin/security-access/api-keys): key setup for the fallback provider.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): the paper behind a common parameter-efficient fine-tuning approach.
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index): practical adapter and parameter-efficient fine-tuning tooling.
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/main/sft_trainer): supervised fine-tuning trainer documentation.
- [Mistral fine-tuning documentation](https://docs.mistral.ai/capabilities/finetuning/text_vision_finetuning/): hosted provider fine-tuning workflow and data expectations.
- [OpenAI fine-tuning data formatting](https://help.openai.com/en/articles/6811186-how-do-i-format-my-fine-tuning-data-for-the-openai-api): provider guidance for supervised fine-tuning examples.
