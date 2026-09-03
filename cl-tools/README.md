# 📊 FRAG Evaluation Guide

This guide explains how to evaluate your RAG pipeline using **RAGAS** metrics. You can measure how well your system retrieves relevant documents and generates accurate answers, and compare different configurations to find the best setup for your data.

---

## Table of Contents

1. [What is Evaluation and Why Do It?](#1-what-is-evaluation-and-why-do-it)
2. [Quick Start (5 Minutes)](#2-quick-start-5-minutes)
3. [Installation](#3-installation)
4. [Creating a Test Dataset](#4-creating-a-test-dataset)
5. [Generating a Test Dataset Automatically](#5-generating-a-test-dataset-automatically)
6. [Running an Evaluation](#6-running-an-evaluation)
7. [Comparing Configurations](#7-comparing-configurations)
8. [Understanding the Metrics](#8-understanding-the-metrics)
9. [Viewing Past Results](#9-viewing-past-results)
10. [Advanced Usage](#10-advanced-usage)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What is Evaluation and Why Do It?

When you build a RAG system, you have many settings to choose from:

- **Chunking method**: sentence, semantic, or token-based?
- **Chunk size**: 20 sentences? 60? 512 tokens?
- **Number of retrieved documents**: 3? 5? 10?
- **HyDE**: does generating a hypothetical answer before retrieval help?
- **Hybrid search**: does adding BM25 keyword search improve results?

Instead of guessing which combination works best, you can **measure** each configuration against a set of test questions with known answers. The evaluation tool runs your questions through the RAG pipeline and uses an LLM judge ([RAGAS](https://docs.ragas.io/)) to score the results on four dimensions:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved documents? (No hallucinations) |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Precision** | Are the top-ranked retrieved documents the most useful ones? |
| **Context Recall** | Was all the information needed to answer the question retrieved? |

---

## 2. Quick Start (5 Minutes)

Already have your vector database indexed? Here's the fastest path:

### Step 1: Generate a test set automatically
```bash
python cl-tools/evaluate.py generate-testset \
    --num-questions 10 \
    --output eval_data/my_testset.json
```

### Step 2: Run the evaluation
```bash
python cl-tools/evaluate.py run --testset eval_data/my_testset.json
```

### Step 3: Read the results
The tool prints a table showing your scores. Results are also saved to `eval_results/` for later reference.

That's it! Read on for more details.

---

## 3. Installation

Install the additional dependency:

```bash
pip install ragas
```

Or update all dependencies at once:

```bash
pip install -r requirements.txt
```

> **Note**: RAGAS uses an "LLM-as-a-judge" approach. It reuses whatever LLM you already configured in `config/llm_config.py` (Ollama or OpenAI-compatible API). No separate API key is needed.

---

## 4. Creating a Test Dataset

A test dataset is a file containing questions about your data, each paired with the correct answer. You need to create this yourself (or generate it automatically — see Section 5).

### JSON Format

Create a file like `eval_data/my_testset.json`:

```json
[
  {
    "question": "What year was the university founded?",
    "ground_truth": "The university was founded in 1834."
  },
  {
    "question": "What is the maximum student enrollment?",
    "ground_truth": "The maximum enrollment capacity is 18,000 students."
  },
  {
    "question": "Who is the current president?",
    "ground_truth": "The current president is Dr. Jane Smith, appointed in 2021."
  }
]
```

**Rules**:
- The file must be a JSON array of objects.
- Each object must have `"question"` and `"ground_truth"` fields.
- Answers should be complete sentences, matching how your documents state the facts.

### CSV Format

Alternatively, create a CSV file like `eval_data/my_testset.csv`:

```csv
question,ground_truth
"What year was the university founded?","The university was founded in 1834."
"What is the maximum student enrollment?","The maximum enrollment capacity is 18,000 students."
"Who is the current president?","The current president is Dr. Jane Smith, appointed in 2021."
```

**Flexible column names**: The tool accepts these column names interchangeably:
- For questions: `question`, `query`, or `user_input`
- For answers: `ground_truth`, `reference`, `answer`, or `expected_answer`

### How Many Questions?

- **Minimum**: 5–10 questions for a quick sanity check
- **Recommended**: 20–50 questions for meaningful comparisons
- **Thorough**: 100+ questions for statistically significant results

### Tips for Good Test Questions

- ✅ Cover different topics in your document collection
- ✅ Include both simple factual questions and complex multi-part ones
- ✅ Make sure the ground truth answer is actually in your documents
- ❌ Don't ask questions that your documents can't answer
- ❌ Don't make answers too short ("Yes") — write complete sentences

---

## 5. Generating a Test Dataset Automatically

If you don't have test questions ready, the tool can generate them from your indexed documents using the configured LLM.

### Basic Usage

```bash
python cl-tools/evaluate.py generate-testset \
    --num-questions 20 \
    --output eval_data/auto_testset.json
```

### How It Works

1. The tool randomly samples chunks from your ChromaDB collection
2. For each chunk, it asks the LLM to generate question-answer pairs based on that chunk's content
3. The QA pairs are saved to the output file

### Options

```bash
python cl-tools/evaluate.py generate-testset \
    --num-questions 30 \                    # Number of QA pairs (default: 20)
    --output eval_data/auto_testset.csv \   # Save as CSV instead of JSON
    --collection my_other_collection        # Use a different collection
```

### After Generating

**Always review the generated test set!** Open the file and:
- Remove any nonsensical or duplicate questions
- Fix any incorrect ground truth answers
- Add questions that cover important topics the LLM missed

The generated test set is a starting point, not a finished product.

---

## 6. Running an Evaluation

### Evaluate with Current Settings

This runs your test questions through the RAG pipeline using whatever is currently configured in `config/embedding_config.py` and `config/llm_config.py`:

```bash
python cl-tools/evaluate.py run --testset eval_data/my_testset.json
```

### What Happens During Evaluation

For each question in your test set, the tool:

1. **Retrieves** documents from ChromaDB (using your embedding model)
2. **Generates** an answer using the LLM (using your prompt template)
3. **Scores** the result with RAGAS metrics (using the LLM as a judge)

### Output

The tool prints a comparison table:

```
----------------------------------------------------------------------
 Config          | Faithfulness | Answer Relevancy | Context Precision | Context Recall |     Avg     
----------------------------------------------------------------------
 current         |    0.8200    |      0.7800      |       0.7100      |     0.6500     |    0.7400   
----------------------------------------------------------------------

  Best overall: current (avg: 0.7400)

Results saved to: eval_results/2026-09-02_14-30-00_results.json
```

### Show Per-Question Details

Add `--detail` to see how each individual question scored:

```bash
python cl-tools/evaluate.py run --testset eval_data/my_testset.json --detail
```

This shows which questions your RAG system handles well and which ones it struggles with — very useful for debugging.

---

## 7. Comparing Configurations

This is the most powerful feature. You can test different RAG settings side-by-side to find what works best.

### Syntax

Use `--config NAME` to name a configuration, followed by `--set key=value` flags to override specific settings. Repeat for each config you want to test.

### Example: Does HyDE Help?

```bash
python cl-tools/evaluate.py run --testset eval_data/my_testset.json \
    --config baseline \
    --config with_hyde --set use_hyde=true
```

This runs the test set twice:
- `baseline`: uses current settings (HyDE off by default)
- `with_hyde`: same as baseline but with HyDE enabled

### Example: Compare Chunk Sizes

```bash
python cl-tools/evaluate.py run --testset eval_data/my_testset.json \
    --config small_chunks --set chunk_size=20 --set overlap_size=3 \
    --config medium_chunks --set chunk_size=40 --set overlap_size=5 \
    --config large_chunks --set chunk_size=80 --set overlap_size=10
```

> ⚠️ **Important**: Configs that change chunking settings (chunk_size, chunking_method, etc.) will only work correctly if you have already re-indexed your documents with those settings. The evaluation tool uses the existing ChromaDB index — it does NOT re-index for you. To test different chunking configs, you need to:
> 1. Change `config/embedding_config.py`
> 2. Run `python embedding/vector_db_setup.py`
> 3. Then run the evaluation

Configs that only change **retrieval-time** settings work instantly with the existing index:
- `n_results` (how many docs to retrieve)
- `use_hyde` (hypothetical document embeddings)
- `use_hybrid_search` (BM25 + vector fusion)

### Example: Compare Retrieval Strategies

```bash
python cl-tools/evaluate.py run --testset eval_data/my_testset.json \
    --config vector_only \
    --config with_hyde --set use_hyde=true \
    --config with_hybrid --set use_hybrid_search=true \
    --config full --set use_hyde=true --set use_hybrid_search=true
```

### Example: Compare Number of Retrieved Documents

```bash
python cl-tools/evaluate.py run --testset eval_data/my_testset.json \
    --config top3 --set n_results=3 \
    --config top5 --set n_results=5 \
    --config top10 --set n_results=10 \
    --config top20 --set n_results=20
```

### Available Settings to Override

Any setting from `config/embedding_config.py` or `config/llm_config.py` can be overridden:

| Setting | Config File | What it controls |
|---|---|---|
| `chunk_size` | embedding_config | Sentences per chunk (requires re-index) |
| `overlap_size` | embedding_config | Sentence overlap between chunks (requires re-index) |
| `chunking_method` | embedding_config | `sentence`, `semantic`, or `token` (requires re-index) |
| `n_results` | embedding_config | Number of documents retrieved per query |
| `use_hybrid_search` | embedding_config | Enable BM25 + vector fusion |
| `use_hyde` | llm_config | Enable Hypothetical Document Embeddings |
| `llm_model` | llm_config | Ollama model name |
| `openai_model` | llm_config | OpenAI-compatible model name |
| `use_openai` | llm_config | Switch between Ollama and OpenAI API |

---

## 8. Understanding the Metrics

### Faithfulness (0.0 – 1.0)
**"Is the answer supported by the retrieved documents?"**

- **1.0** = Every claim in the answer can be found in the retrieved context
- **0.0** = The answer contains entirely hallucinated information
- **Low score?** Your LLM is making things up. Try a better prompt, a different model, or retrieve more documents.

### Answer Relevancy (0.0 – 1.0)
**"Does the answer actually address the question?"**

- **1.0** = The answer directly and completely addresses the question
- **0.0** = The answer is about something completely different
- **Low score?** Your prompt template might not be guiding the LLM effectively, or the retrieved documents are off-topic.

### Context Precision (0.0 – 1.0)
**"Are the most useful documents ranked at the top?"**

- **1.0** = The most relevant documents are ranked first
- **0.0** = Relevant documents are buried below irrelevant ones
- **Low score?** Try a different embedding model, enable hybrid search, or experiment with HyDE.

### Context Recall (0.0 – 1.0)
**"Was all the information needed to answer the question retrieved?"**

- **1.0** = All necessary information was in the retrieved context
- **0.0** = Critical information was missed entirely
- **Low score?** Try retrieving more documents (`n_results`), enable hybrid search, or check if your chunking is splitting relevant content across chunks.

---

## 9. Viewing Past Results

### List All Results

```bash
python cl-tools/evaluate.py results
```

Output:
```
  Evaluation results in: eval_results/

  #    Date                  Questions    Configs  File
  ----------------------------------------------------------------------
  1    2026-09-02 14:30:00          20         3  2026-09-02_14-30-00_results.json
  2    2026-09-01 10:15:00          10         1  2026-09-01_10-15-00_results.json
```

### View a Specific Result

```bash
python cl-tools/evaluate.py results \
    --file eval_results/2026-09-02_14-30-00_results.json
```

### View Per-Question Breakdown

```bash
python cl-tools/evaluate.py results \
    --file eval_results/2026-09-02_14-30-00_results.json \
    --detail
```

### View Detail for One Config Only

```bash
python cl-tools/evaluate.py results \
    --file eval_results/2026-09-02_14-30-00_results.json \
    --detail --config-name with_hyde
```

---

## 10. Advanced Usage

### Use Specific RAGAS Metrics

By default, all four metrics are computed. To use only some:

```bash
python cl-tools/evaluate.py run \
    --testset eval_data/my_testset.json \
    --metrics faithfulness,answer_relevancy
```

### Save Results to a Custom Directory

```bash
python cl-tools/evaluate.py run \
    --testset eval_data/my_testset.json \
    --output-dir my_experiments/round_3/
```

### Use a Different LLM as the Judge

By default, the evaluation uses whatever LLM is configured in `config/llm_config.py` as the RAGAS judge. You can override this in two ways:

**Option 1: CLI flag (recommended for quick tests)**
```bash
python cl-tools/evaluate.py run \
    --testset eval_data/my_testset.json \
    --judge-model gpt-4o-mini
```

You can also specify a different API endpoint for the judge:
```bash
python cl-tools/evaluate.py run \
    --testset eval_data/my_testset.json \
    --judge-model gpt-4o-mini \
    --judge-base-url https://api.openai.com/v1
```

**Option 2: Config file (for permanent change)**

Edit `config/eval_config.py`:
```python
judge_model = "gpt-4o"  # or any model name available on your LLM backend
```

### Evaluation with OpenAI API vs Ollama

The evaluation tool automatically uses whatever LLM backend you have configured:
- If `use_openai = True` in `config/llm_config.py` → uses the OpenAI-compatible API
- If `use_openai = False` → uses Ollama

The same applies to embeddings:
- If `use_openai_embeddings = True` → uses OpenAI embeddings API
- If `use_openai_embeddings = False` → uses local SentenceTransformer

---

## 11. Troubleshooting

### "Collection 'X' is empty" when generating test set
You need to index your documents first:
```bash
python embedding/vector_db_setup.py
```

### "ragas not installed"
Install it:
```bash
pip install ragas
```

### Evaluation is very slow
Each question requires:
1. One embedding call (retrieval)
2. One LLM call (answer generation)
3. Multiple LLM calls (RAGAS scoring — typically 2–4 per metric per question)

For a 20-question test set with 4 metrics, that's roughly **200+ LLM calls**. With a local Ollama model, this may take 10–30 minutes. With a fast API, 2–5 minutes.

**Tips to speed up**:
- Reduce the number of questions in your test set
- Use fewer metrics: `--metrics faithfulness,context_recall`
- Use a fast LLM model for the judge

### Scores are all very low
- Check that your test questions are actually answerable by your documents
- Check that the ground truth answers match what's in your documents
- Try retrieving more documents: `--set n_results=10`
- Review the per-question breakdown with `--detail` to see what's going wrong

### "Unknown config key" warning
You may have a typo in a `--set` flag. Check the setting name matches exactly what's in `config/embedding_config.py` or `config/llm_config.py`.

### JSON parsing errors / InstructorRetryException / IncompleteOutputException

This is the **most common issue** when using "thinking" models (like Qwen3, QwQ, DeepSeek-R1, or any model that produces `reasoning_content`) as the RAGAS judge.

**What happens**: RAGAS uses `instructor` to extract structured JSON from the judge LLM. Thinking models put extensive reasoning into a separate `reasoning_content` field, which consumes most of the output token budget. The actual JSON content then gets truncated or garbled, producing errors like:
```
Invalid JSON: key must be a string at line 2 column 1
IncompleteOutputException: The output is incomplete due to a max_tokens length limit.
```

**Solution 1 — Use a non-thinking judge model** (recommended):
```bash
# Use a different model that doesn't use thinking tokens
python cl-tools/evaluate.py run \
    --testset eval_data/my_testset.json \
    --judge-model gpt-4o-mini
```

If your GPUStack/Ollama has multiple models, pick one that doesn't do chain-of-thought reasoning in a separate field.

**Solution 2 — Skip the problematic metric**:

`answer_relevancy` is the metric that fails most often with thinking models. You can skip it:
```bash
python cl-tools/evaluate.py run \
    --testset eval_data/my_testset.json \
    --metrics faithfulness,context_precision,context_recall
```

**Solution 3 — Increase max_tokens on your LLM server**:

If you control the LLM server (GPUStack, vLLM, etc.), increase the max output tokens setting. The thinking model needs enough budget for both reasoning AND the JSON output.

> **Note**: The evaluation tool now scores each metric independently. If one metric fails (e.g., `answer_relevancy`), the other metrics will still be computed and reported. You'll see `N/A` for the failed metric instead of a crash.

---

## Complete Command Reference

```
python cl-tools/evaluate.py run
    --testset PATH          Path to test set (.json or .csv) [required]
    --config NAME           Name a config profile (repeat for multiple)
    --set KEY=VALUE         Override a setting for the current config
    --metrics NAMES         Comma-separated metric names
    --output-dir DIR        Where to save results
    --detail                Show per-question scores
    --judge-model NAME      Override the LLM judge model
    --judge-base-url URL    Override the judge LLM API base URL

python cl-tools/evaluate.py generate-testset
    --num-questions N       Number of QA pairs to generate (default: 20)
    --output PATH           Output file path (.json or .csv) [required]
    --collection NAME       ChromaDB collection to sample from

python cl-tools/evaluate.py results
    --file PATH             Specific result file to display
    --detail                Show per-question scores
    --config-name NAME      Filter detail to one config
    --output-dir DIR        Directory containing result files
```
