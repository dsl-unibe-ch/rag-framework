"""Core evaluation engine.

Runs a test dataset through the RAG pipeline for one or more configuration
profiles and scores the results with RAGAS metrics.

The runner reuses the existing retriever classes (:class:`ChromaRetriever`,
:class:`OpenAIChromaRetriever`) and responder classes (:class:`Responder`,
:class:`OpenAIResponder`) so evaluation exercises the exact same code paths
as the production chat and search features.
"""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

from evaluation.testset import TestSample
from evaluation.report import (
    EvalReport,
    ConfigResult,
    QuestionResult,
)


# -----------------------------------------------------------------------
# Config override helpers
# -----------------------------------------------------------------------

# Keys that live in embedding_config
_EMBEDDING_KEYS = {
    "model_name", "use_openai_embeddings", "openai_embedding_model",
    "openai_embedding_base_url", "vector_db", "collection_name", "raw_db",
    "data_language", "db_directory", "chunk_size", "overlap_size",
    "chunking_method", "semantic_breakpoint_percentile",
    "semantic_buffer_size", "semantic_max_chunk_sentences",
    "token_chunk_size", "token_chunk_overlap", "token_encoding",
    "embedding_batch_size", "n_results", "source_url",
    "use_hybrid_search", "hybrid_rrf_k", "hybrid_candidates",
}

# Keys that live in llm_config
_LLM_KEYS = {
    "llm_model", "use_openai", "openai_model", "openai_base_url",
    "prompt", "record_data", "use_hyde",
}


def _apply_overrides(overrides: Dict) -> Dict:
    """Apply config overrides and return a dict of original values.

    Temporarily patches the module-level variables in
    ``config.embedding_config`` and ``config.llm_config``.
    """
    import config.embedding_config as ec
    import config.llm_config as lc

    originals: Dict = {}
    for key, value in overrides.items():
        if key in _EMBEDDING_KEYS:
            originals[key] = getattr(ec, key)
            setattr(ec, key, value)
        elif key in _LLM_KEYS:
            originals[key] = getattr(lc, key)
            setattr(lc, key, value)
        else:
            print(f"  Warning: unknown config key '{key}' — skipped.")
    return originals


def _restore_overrides(originals: Dict) -> None:
    """Restore config values from a dict returned by :func:`_apply_overrides`."""
    import config.embedding_config as ec
    import config.llm_config as lc

    for key, value in originals.items():
        if key in _EMBEDDING_KEYS:
            setattr(ec, key, value)
        elif key in _LLM_KEYS:
            setattr(lc, key, value)


# -----------------------------------------------------------------------
# RAGAS LLM judge setup
# -----------------------------------------------------------------------

def _build_ragas_llm(use_openai: bool, model: str, base_url: str,
                     api_key: Optional[str]):
    """Build a RAGAS-compatible LLM wrapper for scoring.

    Uses ``ragas.llms.llm_factory`` with the user's configured LLM
    backend so no additional API keys are needed.

    .. note::

       "Thinking" models (e.g. QwQ, Qwen3 with ``think=True``) are
       problematic as RAGAS judges because their reasoning tokens
       consume the output budget, leaving too few tokens for the
       structured JSON that RAGAS/instructor expects.  If you hit
       ``InstructorRetryException`` or ``IncompleteOutputException``,
       switch to a non-thinking judge model via ``--judge-model``.
    """
    from ragas.llms import llm_factory
    from openai import OpenAI as _OpenAI

    if use_openai:
        client = _OpenAI(api_key=api_key, base_url=base_url)
    else:
        # Ollama exposes an OpenAI-compatible endpoint at /v1
        client = _OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )

    # ------------------------------------------------------------------
    # HOTFIX for Thinking Models:
    # We patch the client to automatically strip <think>...</think> tags 
    # from the response content. We also inject a high max_tokens limit 
    # to prevent reasoning tokens from exhausting the budget before the 
    # required JSON is generated.
    # ------------------------------------------------------------------
    original_create = client.chat.completions.create
    
    def _patched_create(*args, **kwargs):
        # Force a high token limit to accommodate reasoning + JSON
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 8192

        resp = original_create(*args, **kwargs)
        
        # Strip <think> tags from the output so `instructor`'s JSON 
        # parser doesn't choke on them.
        import re
        if hasattr(resp, "choices"):
            for choice in resp.choices:
                if getattr(choice, "message", None) and getattr(choice.message, "content", None):
                    # Remove <think>...</think> including newlines
                    clean_content = re.sub(
                        r'<think>.*?</think>', '', 
                        choice.message.content, 
                        flags=re.DOTALL
                    ).strip()
                    choice.message.content = clean_content

        return resp

    client.chat.completions.create = _patched_create

    return llm_factory(model, provider="openai", client=client)


# -----------------------------------------------------------------------
# Single-config evaluation
# -----------------------------------------------------------------------

def _evaluate_config(
    config_name: str,
    overrides: Dict,
    samples: List[TestSample],
    metric_names: List[str],
    judge_llm,
    judge_embeddings,
    env_path: str,
) -> ConfigResult:
    """Run the RAG pipeline and RAGAS evaluation for one config profile.

    Args:
        config_name: Human-readable label for this config.
        overrides: Config overrides to apply before the run.
        samples: The test questions + ground truths.
        metric_names: RAGAS metric names to compute.
        judge_llm: The RAGAS LLM wrapper for scoring.
        env_path: Path to the ``.env`` file.

    Returns:
        A :class:`ConfigResult` with aggregate and per-question scores.
    """
    originals = _apply_overrides(overrides)

    try:
        return _run_pipeline_and_score(
            config_name, overrides, samples, metric_names, judge_llm,
            judge_embeddings, env_path,
        )
    finally:
        _restore_overrides(originals)


def _run_pipeline_and_score(
    config_name: str,
    overrides: Dict,
    samples: List[TestSample],
    metric_names: List[str],
    judge_llm,
    judge_embeddings,
    env_path: str,
) -> ConfigResult:
    """Internal implementation: retrieve + generate + score."""
    # Re-import after overrides are applied so we read the patched values.
    import config.embedding_config as ec
    import config.llm_config as lc
    from retrieval.main import ChromaRetriever, OpenAIChromaRetriever
    from retrieval.hyde import (
        generate_hypothetical_document,
        generate_hypothetical_document_ollama,
    )
    from llm.main import Responder, OpenAIResponder

    load_dotenv(env_path)
    api_key = os.environ.get("OPENAI_API_KEY")

    # --- Build retriever ---
    if ec.use_openai_embeddings:
        from openai import OpenAI as _OpenAI
        embed_client = _OpenAI(
            api_key=api_key,
            base_url=ec.openai_embedding_base_url,
        )
        retriever = OpenAIChromaRetriever(
            openai_client=embed_client,
            embedding_model=ec.openai_embedding_model,
            db_path=ec.db_directory,
            db_collection=ec.collection_name,
            n_results=ec.n_results,
        )
    else:
        retriever = ChromaRetriever(
            embedding_model=ec.model_name,
            db_path=ec.db_directory,
            db_collection=ec.collection_name,
            n_results=ec.n_results,
        )

    # --- Process each question ---
    question_results: List[QuestionResult] = []

    for i, sample in enumerate(samples):
        print(f"    Q{i + 1}/{len(samples)}: {sample.question[:80]}...")

        # Optional HyDE
        hyde_doc = None
        if lc.use_hyde:
            if lc.use_openai:
                from openai import OpenAI as _OpenAI
                hyde_client = _OpenAI(
                    api_key=api_key, base_url=lc.openai_base_url,
                )
                hyde_doc = generate_hypothetical_document(
                    sample.question, hyde_client, lc.openai_model,
                )
            else:
                hyde_doc = generate_hypothetical_document_ollama(
                    sample.question, lc.llm_model,
                )

        # Retrieve
        search_results = retriever.retrieve(
            sample.question,
            embed_text=hyde_doc,
            use_hybrid=ec.use_hybrid_search,
        )
        formatted_data = retriever.format_results_for_prompt(search_results)

        # Extract raw context strings for RAGAS
        contexts: List[str] = []
        if search_results and "documents" in search_results:
            contexts = search_results["documents"][0]

        # Generate answer
        if lc.use_openai:
            from openai import OpenAI as _OpenAI
            llm_client = _OpenAI(
                api_key=api_key, base_url=lc.openai_base_url,
            )
            responder = OpenAIResponder(
                data=formatted_data,
                model=lc.openai_model,
                prompt_template=lc.prompt,
                query=sample.question,
                client=llm_client,
            )
        else:
            responder = Responder(
                data=formatted_data,
                model=lc.llm_model,
                prompt_template=lc.prompt,
                query=sample.question,
            )

        try:
            answer = responder.generate_response()
        except Exception as exc:
            print(f"      LLM generation failed: {exc}")
            answer = "[Generation failed]"

        question_results.append(QuestionResult(
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=answer,
            retrieved_contexts=contexts,
            scores={},  # filled in by RAGAS below
        ))

    # --- Score with RAGAS ---
    print(f"  Scoring {len(question_results)} answers with RAGAS...")
    aggregate, per_q_scores = _score_with_ragas(
        question_results, metric_names, judge_llm, judge_embeddings,
    )

    # Merge per-question scores back into results.
    for qr, q_scores in zip(question_results, per_q_scores):
        qr.scores = q_scores

    return ConfigResult(
        config_name=config_name,
        overrides=overrides,
        aggregate_scores=aggregate,
        per_question=question_results,
    )


def _score_with_ragas(
    question_results: List[QuestionResult],
    metric_names: List[str],
    judge_llm,
    judge_embeddings,
) -> tuple:
    """Run RAGAS evaluation and return (aggregate_dict, per_question_list).

    Each metric is evaluated **independently** so that a failure in one
    metric (e.g. the judge model producing invalid JSON for
    ``answer_relevancy``) does not prevent the other metrics from being
    reported.

    Returns:
        A tuple of ``(aggregate_scores, per_question_scores)`` where
        ``aggregate_scores`` is a dict mapping metric names to floats and
        ``per_question_scores`` is a list of dicts (one per question).
    """
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    import ragas.metrics as ragas_metrics

    # Resolve metric objects from names.
    valid_metrics: List[tuple] = []  # (name, metric_obj)
    for name in metric_names:
        metric_obj = getattr(ragas_metrics, name, None)
        if metric_obj is None:
            print(f"  Warning: unknown RAGAS metric '{name}' — skipped.")
            continue
        
        # FIX: RAGAS 0.4.x answer_relevancy requests 3 generations (strictness=3) by default.
        # When using instructor for structured outputs, it flattens the output to 1 generation,
        # which causes massive warning spam ("LLM returned 1 generations instead of requested 3").
        if name == "answer_relevancy" and hasattr(metric_obj, "strictness"):
            metric_obj.strictness = 1
            
        valid_metrics.append((name, metric_obj))

    if not valid_metrics:
        print("  No valid metrics to evaluate.")
        return {}, [{} for _ in question_results]

    # Build RAGAS dataset.
    ragas_samples = []
    for qr in question_results:
        ragas_samples.append(SingleTurnSample(
            user_input=qr.question,
            retrieved_contexts=qr.retrieved_contexts or [""],
            response=qr.answer,
            reference=qr.ground_truth,
        ))

    dataset = EvaluationDataset(samples=ragas_samples)

    # ------------------------------------------------------------------
    # Evaluate each metric independently for resilience.
    # Thinking models (QwQ, Qwen3, etc.) often fail on specific metrics
    # (especially answer_relevancy) while succeeding on others.
    # ------------------------------------------------------------------
    aggregate: Dict[str, Optional[float]] = {}
    per_q_scores: List[Dict[str, Optional[float]]] = [
        {} for _ in question_results
    ]

    for metric_name, metric_obj in valid_metrics:
        print(f"    Scoring metric: {metric_name}...")
        try:
            results = ragas_evaluate(
                dataset=dataset,
                metrics=[metric_obj],
                llm=judge_llm,
                embeddings=judge_embeddings,
            )

            # Extract per-question scores and compute aggregate.
            try:
                df = results.to_pandas()
                col_vals = []
                for idx, (_, row) in enumerate(df.iterrows()):
                    val = row.get(metric_name)
                    if val is not None:
                        try:
                            f_val = float(val)
                            per_q_scores[idx][metric_name] = f_val
                            import math
                            if not math.isnan(f_val):
                                col_vals.append(f_val)
                        except (TypeError, ValueError):
                            per_q_scores[idx][metric_name] = None
                    else:
                        per_q_scores[idx][metric_name] = None
                
                if col_vals:
                    aggregate[metric_name] = sum(col_vals) / len(col_vals)
                else:
                    aggregate[metric_name] = None
            except Exception:
                for idx in range(len(question_results)):
                    per_q_scores[idx][metric_name] = None
                aggregate[metric_name] = None

            score_str = f"{aggregate[metric_name]:.4f}" if aggregate[metric_name] is not None else "N/A"
            print(f"      {metric_name}: {score_str}")

        except Exception as exc:
            import traceback
            short_err = str(exc)[:200]
            print(f"      {metric_name}: FAILED — {short_err}")
            print(f"      Traceback: {traceback.format_exc()}")
            aggregate[metric_name] = None
            for idx in range(len(question_results)):
                per_q_scores[idx][metric_name] = None

    return aggregate, per_q_scores


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def run_evaluation(
    testset_path: str,
    samples: List[TestSample],
    configs: Dict[str, Dict],
    metric_names: List[str],
    env_path: str,
    config_filter: Optional[List[str]] = None,
    judge_model_override: Optional[str] = None,
    judge_base_url_override: Optional[str] = None,
) -> EvalReport:
    """Run the full evaluation pipeline.

    Args:
        testset_path: Path to the test-set file (for the report).
        samples: Loaded test samples.
        configs: Dict mapping config names to override dicts.
        metric_names: RAGAS metric names to compute.
        env_path: Path to the ``.env`` file.
        config_filter: If set, only run these config names.
        judge_model_override: If set, use this model as the RAGAS judge
            instead of the one in eval_config / llm_config.
        judge_base_url_override: If set, use this base URL for the judge.

    Returns:
        An :class:`EvalReport` with results for all requested configs.
    """
    import config.llm_config as lc
    import config.eval_config as evc
    import config.embedding_config as ec

    load_dotenv(env_path)
    api_key = os.environ.get("OPENAI_API_KEY")

    # Determine the judge LLM model.
    judge_model_name = (
        judge_model_override
        or evc.judge_model
        or (lc.openai_model if lc.use_openai else lc.llm_model)
    )
    judge_base_url = (
        judge_base_url_override
        or (lc.openai_base_url if lc.use_openai else None)
    )

    print(f"\n--- RAGAS Evaluation ---")
    print(f"  Test set    : {testset_path} ({len(samples)} questions)")
    print(f"  Metrics     : {', '.join(metric_names)}")
    print(f"  Judge LLM   : {judge_model_name}")
    print(f"  Configs     : {', '.join(configs.keys())}")

    # Build the RAGAS judge LLM.
    judge_llm = _build_ragas_llm(
        use_openai=lc.use_openai,
        model=judge_model_name,
        base_url=judge_base_url or "http://localhost:11434/v1",
        api_key=api_key,
    )

    # Build RAGAS embeddings (needed for answer_relevancy metric).
    judge_embeddings = _build_ragas_embeddings(ec, api_key)

    report = EvalReport(
        testset_path=testset_path,
        num_questions=len(samples),
        metrics=metric_names,
    )

    # Filter configs if requested.
    run_configs = configs
    if config_filter:
        run_configs = {k: v for k, v in configs.items() if k in config_filter}
        missing = set(config_filter) - set(run_configs.keys())
        if missing:
            print(f"  Warning: config(s) not found: {missing}")

    for i, (name, overrides) in enumerate(run_configs.items()):
        print(f"\n[{i + 1}/{len(run_configs)}] Evaluating config: {name}")
        if overrides:
            print(f"  Overrides: {overrides}")

        result = _evaluate_config(
            config_name=name,
            overrides=overrides,
            samples=samples,
            metric_names=metric_names,
            judge_llm=judge_llm,
            judge_embeddings=judge_embeddings,
            env_path=env_path,
        )
        report.configs.append(result)

    return report


def _build_ragas_embeddings(ec, api_key: Optional[str]):
    """Build RAGAS-compatible embeddings from the embedding config.

    Required by metrics like ``answer_relevancy`` that need to embed
    generated questions for comparison.
    """
    try:
        if ec.use_openai_embeddings:
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            # Use LangChain's wrapper because it reliably provides embed_query
            lc_emb = OpenAIEmbeddings(
                model=ec.openai_embedding_model,
                openai_api_base=ec.openai_embedding_base_url,
                openai_api_key=api_key
            )
            return LangchainEmbeddingsWrapper(lc_emb)
        else:
            from ragas.embeddings import HuggingfaceEmbeddings
            return HuggingfaceEmbeddings(model_name=ec.model_name)
    except Exception as exc:
        print(f"  Warning: could not build RAGAS embeddings: {exc}")
        print(f"  Metrics requiring embeddings (answer_relevancy) may fail.")
        return None
