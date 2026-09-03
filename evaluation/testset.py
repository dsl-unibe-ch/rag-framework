"""Test dataset loading, validation, and synthetic generation.

Supports two input formats:

* **JSON** -- a list of objects, each with ``"question"`` and
  ``"ground_truth"`` keys.
* **CSV** -- a file with ``question`` and ``ground_truth`` columns
  (header row required).

The :func:`generate_synthetic_testset` function uses the configured LLM to
create question-answer pairs from documents already stored in ChromaDB.
"""

import csv
import json
import os
import random
from typing import List, Optional

from dataclasses import dataclass


@dataclass
class TestSample:
    """A single evaluation sample: a question and its expected answer."""

    question: str
    ground_truth: str


# -----------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------

def load_testset(path: str) -> List[TestSample]:
    """Load a test dataset from a JSON or CSV file.

    Args:
        path: Path to the test-set file.

    Returns:
        A list of :class:`TestSample` objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file format is unsupported or the schema is
            invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test-set file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return _load_json(path)
    if ext == ".csv":
        return _load_csv(path)

    raise ValueError(
        f"Unsupported test-set format '{ext}'.  Use .json or .csv."
    )


def _load_json(path: str) -> List[TestSample]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(
            "JSON test set must be a list of objects, e.g. "
            '[{"question": "...", "ground_truth": "..."}]'
        )

    samples: List[TestSample] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not an object: {item!r}")
        q = item.get("question", "").strip()
        gt = item.get("ground_truth", "").strip()
        if not q:
            raise ValueError(f"Item {idx} is missing a 'question' field.")
        if not gt:
            raise ValueError(f"Item {idx} is missing a 'ground_truth' field.")
        samples.append(TestSample(question=q, ground_truth=gt))

    if not samples:
        raise ValueError("Test set is empty.")
    return samples


def _load_csv(path: str) -> List[TestSample]:
    samples: List[TestSample] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV file appears to be empty.")

        # Allow flexible column names
        col_map = {}
        for name in reader.fieldnames:
            lower = name.strip().lower().replace(" ", "_")
            if lower in ("question", "query", "user_input"):
                col_map["question"] = name
            elif lower in ("ground_truth", "groundtruth", "reference",
                           "expected_answer", "answer"):
                col_map["ground_truth"] = name

        if "question" not in col_map:
            raise ValueError(
                "CSV must have a 'question' column (or 'query' / 'user_input').  "
                f"Found columns: {reader.fieldnames}"
            )
        if "ground_truth" not in col_map:
            raise ValueError(
                "CSV must have a 'ground_truth' column (or 'reference' / "
                f"'answer').  Found columns: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader):
            q = row.get(col_map["question"], "").strip()
            gt = row.get(col_map["ground_truth"], "").strip()
            if not q:
                raise ValueError(f"Row {idx + 2} has an empty question.")
            if not gt:
                raise ValueError(f"Row {idx + 2} has an empty ground_truth.")
            samples.append(TestSample(question=q, ground_truth=gt))

    if not samples:
        raise ValueError("CSV test set is empty.")
    return samples


# -----------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------

def save_testset(samples: List[TestSample], path: str) -> None:
    """Save a test dataset to a JSON or CSV file.

    The format is inferred from the file extension.

    Args:
        samples: The test samples to save.
        path: Destination file path (.json or .csv).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["question", "ground_truth"])
            writer.writeheader()
            for s in samples:
                writer.writerow({"question": s.question,
                                 "ground_truth": s.ground_truth})
    else:
        # Default to JSON
        data = [{"question": s.question, "ground_truth": s.ground_truth}
                for s in samples]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to: {path}")


# -----------------------------------------------------------------------
# Synthetic generation
# -----------------------------------------------------------------------

_GENERATION_PROMPT = """\
You are helping create an evaluation dataset for a Retrieval-Augmented \
Generation system.  Below is a document excerpt from the knowledge base.

DOCUMENT:
{document}

Based ONLY on the information in the document above, generate exactly \
{count} question-answer pairs.  Each question should be answerable \
using the document text.  The answer must be factual and grounded in \
the document.

Return your response as a JSON array of objects, each with "question" \
and "ground_truth" keys.  Return ONLY the JSON array, no markdown \
fences, no extra text.

Example format:
[
  {{"question": "What year was X founded?", "ground_truth": "X was founded in 1990."}},
  {{"question": "Who is the director of Y?", "ground_truth": "The director of Y is John Smith."}}
]
"""


def generate_synthetic_testset(
    num_questions: int,
    collection_name: str,
    db_directory: str,
    use_openai: bool,
    openai_model: str,
    openai_base_url: str,
    ollama_model: str,
    api_key: Optional[str] = None,
) -> List[TestSample]:
    """Generate a synthetic test set from documents stored in ChromaDB.

    Samples random chunks from the collection and asks the LLM to
    generate question-answer pairs grounded in each chunk.

    Args:
        num_questions: Target number of QA pairs to generate.
        collection_name: ChromaDB collection to sample from.
        db_directory: Path to the ChromaDB database.
        use_openai: Whether to use the OpenAI-compatible API.
        openai_model: Model name for the OpenAI-compatible API.
        openai_base_url: Base URL for the OpenAI-compatible API.
        ollama_model: Model name for Ollama.
        api_key: OpenAI API key (can be ``None`` for Ollama).

    Returns:
        A list of generated :class:`TestSample` objects.
    """
    import chromadb

    client = chromadb.PersistentClient(path=db_directory)
    collection = client.get_collection(name=collection_name)
    total_docs = collection.count()

    if total_docs == 0:
        raise RuntimeError(
            f"Collection '{collection_name}' is empty.  "
            "Index some documents first with vector_db_setup.py."
        )

    # Sample chunks.  We ask for 2–3 QA pairs per chunk, so we need
    # roughly num_questions / 2 chunks.
    n_chunks = max(1, min(total_docs, (num_questions + 1) // 2))
    # Fetch a pool of candidates, then randomly sample from them.
    fetch_count = min(total_docs, max(n_chunks * 3, 50))
    result = collection.get(
        limit=fetch_count,
        include=["documents"],
    )
    all_docs = result.get("documents") or []
    if not all_docs:
        raise RuntimeError("Could not retrieve any documents from the collection.")

    # Shuffle and pick n_chunks distinct docs.
    random.shuffle(all_docs)
    selected_docs = all_docs[:n_chunks]

    samples: List[TestSample] = []
    # How many QA pairs to request per chunk.
    per_chunk = max(1, (num_questions + n_chunks - 1) // n_chunks)

    print(f"Generating ~{num_questions} QA pairs from {n_chunks} sampled chunks...")

    for i, doc_text in enumerate(selected_docs):
        if len(samples) >= num_questions:
            break

        remaining = num_questions - len(samples)
        ask_count = min(per_chunk, remaining)
        prompt = _GENERATION_PROMPT.format(document=doc_text, count=ask_count)

        try:
            raw_response = _call_llm_for_generation(
                prompt=prompt,
                use_openai=use_openai,
                openai_model=openai_model,
                openai_base_url=openai_base_url,
                ollama_model=ollama_model,
                api_key=api_key,
            )
            parsed = _parse_qa_json(raw_response)
            samples.extend(parsed)
            print(f"  Chunk {i + 1}/{n_chunks}: generated {len(parsed)} pairs")
        except Exception as exc:
            print(f"  Chunk {i + 1}/{n_chunks}: generation failed — {exc}")
            continue

    # Trim to the exact target.
    samples = samples[:num_questions]
    print(f"Total generated: {len(samples)} QA pairs")
    return samples


def _call_llm_for_generation(
    prompt: str,
    use_openai: bool,
    openai_model: str,
    openai_base_url: str,
    ollama_model: str,
    api_key: Optional[str],
) -> str:
    """Call the LLM to generate QA pairs and return the raw text response."""
    if use_openai:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=openai_base_url)
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system",
                 "content": "You generate evaluation QA pairs in JSON format."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    else:
        import ollama
        response = ollama.chat(
            model=ollama_model,
            messages=[
                {"role": "system",
                 "content": "You generate evaluation QA pairs in JSON format."},
                {"role": "user", "content": prompt},
            ],
            think=False,
        )
        return response.message.content.strip()


def _parse_qa_json(raw: str) -> List[TestSample]:
    """Parse the LLM's JSON response into TestSample objects.

    Handles common LLM quirks like wrapping JSON in markdown code fences.
    """
    text = raw.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last lines (```json and ```)
        start = 1
        end = len(lines)
        for j in range(len(lines) - 1, 0, -1):
            if lines[j].strip().startswith("```"):
                end = j
                break
        text = "\n".join(lines[start:end]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the JSON array from within the text.
        import re
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse LLM output as JSON: {text[:200]}")

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of QA objects.")

    samples = []
    for item in data:
        q = item.get("question", "").strip()
        gt = item.get("ground_truth", "").strip()
        if q and gt:
            samples.append(TestSample(question=q, ground_truth=gt))
    return samples
