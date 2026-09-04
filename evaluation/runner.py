"""Core evaluation engine.

Runs a test dataset through the RAG pipeline for one or more configuration
profiles and scores the results with RAGAS metrics.
"""

import os
import re
import asyncio
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

_LLM_KEYS = {
    "llm_model", "use_openai", "openai_model", "openai_base_url",
    "prompt", "record_data", "use_hyde",
}


def _apply_overrides(overrides: Dict) -> Dict:
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

def _build_ragas_llm(use_openai: bool, model: str, base_url: str, api_key: Optional[str]):
    """Build a RAGAS-compatible LLM wrapper for scoring."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if not use_openai:
        api_key = "ollama"
        base_url = "http://localhost:11434/v1"

    # Inject provider-specific parameters to disable reasoning models
    model_kwargs = {}
    chat_kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    model_lower = model.lower()
    if "gpt-5" in model_lower or "o1" in model_lower or "o3" in model_lower:
        model_kwargs["reasoning"] = {"effort": "none"}
    elif "minimax" in model_lower:
        chat_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    elif "gpt-oss" in model_lower:
        model_kwargs["reasoning_effort"] = "low"

    if model_kwargs:
        chat_kwargs["model_kwargs"] = model_kwargs

    chat_model = ChatOpenAI(**chat_kwargs)

    ragas_llm = LangchainLLMWrapper(chat_model)

    # Monkey-patch Ragas wrapper to aggressively strip <think> tags from 
    # both sync and async outputs before the JSON parser chokes on them.
    original_generate_text = ragas_llm.generate_text
    def _patched_generate_text(*args, **kwargs):
        result = original_generate_text(*args, **kwargs)
        for gen in result.generations:
            for chunk in gen:
                chunk.text = re.sub(r'<think>.*?</think>', '', chunk.text, flags=re.DOTALL).strip()
        return result
    ragas_llm.generate_text = _patched_generate_text

    original_agenerate_text = ragas_llm.agenerate_text
    async def _patched_agenerate_text(*args, **kwargs):
        result = await original_agenerate_text(*args, **kwargs)
        for gen in result.generations:
            for chunk in gen:
                chunk.text = re.sub(r'<think>.*?</think>', '', chunk.text, flags=re.DOTALL).strip()
        return result
    ragas_llm.agenerate_text = _patched_agenerate_text

    return ragas_llm


def _build_ragas_embeddings(ec, api_key: Optional[str]):
    """Build RAGAS-compatible embeddings supporting async."""
    if ec.use_openai_embeddings:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        lc_emb = OpenAIEmbeddings(
            model=ec.openai_embedding_model,
            base_url=ec.openai_embedding_base_url,
            api_key=api_key
        )
        return LangchainEmbeddingsWrapper(lc_emb)
    else:
        from ragas.embeddings import BaseRagasEmbeddings
        from retrieval.main import _ST_MODEL_CACHE
        from sentence_transformers import SentenceTransformer

        class SharedLocalEmbeddings(BaseRagasEmbeddings):
            """Reuses the existing loaded ST model and implements async methods."""
            def __init__(self, model_name: str):
                # self.model MUST be a string so Ragas telemetry validation doesn't crash
                self.model = model_name 
                if model_name not in _ST_MODEL_CACHE:
                    _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
                # Store the actual object under a different attribute
                self.st_model = _ST_MODEL_CACHE[model_name]

            def embed_query(self, text: str) -> list[float]:
                return self.st_model.encode(text).tolist()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.st_model.encode(texts).tolist()

            async def aembed_query(self, text: str) -> list[float]:
                return await asyncio.to_thread(self.embed_query, text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return await asyncio.to_thread(self.embed_documents, texts)

        return SharedLocalEmbeddings(ec.model_name)


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

    question_results: List[QuestionResult] = []

    for i, sample in enumerate(samples):
        print(f"    Q{i + 1}/{len(samples)}: {sample.question[:80]}...")

        hyde_doc = None
        if lc.use_hyde:
            if lc.use_openai:
                from openai import OpenAI as _OpenAI
                hyde_client = _OpenAI(api_key=api_key, base_url=lc.openai_base_url)
                hyde_doc = generate_hypothetical_document(sample.question, hyde_client, lc.openai_model)
            else:
                hyde_doc = generate_hypothetical_document_ollama(sample.question, lc.llm_model)

        search_results = retriever.retrieve(
            sample.question,
            embed_text=hyde_doc,
            use_hybrid=ec.use_hybrid_search,
        )
        formatted_data = retriever.format_results_for_prompt(search_results)

        contexts: List[str] = []
        if search_results and "documents" in search_results and search_results["documents"]:
            contexts = search_results["documents"][0]

        if lc.use_openai:
            from openai import OpenAI as _OpenAI
            llm_client = _OpenAI(api_key=api_key, base_url=lc.openai_base_url)
            responder = OpenAIResponder(
                data=formatted_data, model=lc.openai_model,
                prompt_template=lc.prompt, query=sample.question, client=llm_client,
            )
        else:
            responder = Responder(
                data=formatted_data, model=lc.llm_model,
                prompt_template=lc.prompt, query=sample.question,
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
            scores={},
        ))

    print(f"  Scoring {len(question_results)} answers with RAGAS...")
    aggregate, per_q_scores = _score_with_ragas(
        question_results, metric_names, judge_llm, judge_embeddings,
    )

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
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.run_config import RunConfig
    import ragas.metrics as ragas_metrics

    valid_metrics = []
    for name in metric_names:
        metric_obj = getattr(ragas_metrics, name, None)
        if metric_obj is None:
            print(f"  Warning: unknown RAGAS metric '{name}' — skipped.")
            continue
        if name == "answer_relevancy" and hasattr(metric_obj, "strictness"):
            metric_obj.strictness = 1
            
        # Explicitly assign llm and embeddings to avoid Ragas fallback bugs
        if hasattr(metric_obj, "llm"):
            metric_obj.llm = judge_llm
        if hasattr(metric_obj, "embeddings"):
            metric_obj.embeddings = judge_embeddings
            
        valid_metrics.append((name, metric_obj))

    if not valid_metrics:
        print("  No valid metrics to evaluate.")
        return {}, [{} for _ in question_results]

    ragas_samples = []
    for qr in question_results:
        # Fallback to prevent tokenization crashes on empty results
        ctx = qr.retrieved_contexts if qr.retrieved_contexts else ["No context retrieved"]
        ragas_samples.append(SingleTurnSample(
            user_input=qr.question,
            retrieved_contexts=ctx,
            response=qr.answer,
            reference=qr.ground_truth,
        ))

    dataset = EvaluationDataset(samples=ragas_samples)
    aggregate = {name: None for name, _ in valid_metrics}
    per_q_scores = [{} for _ in question_results]

    try:
        results = ragas_evaluate(
            dataset=dataset,
            metrics=[m[1] for m in valid_metrics],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(max_workers=2, timeout=600),
            raise_exceptions=False,
        )

        df = results.to_pandas()
        for name, _ in valid_metrics:
            if name in df.columns:
                col_vals = []
                for idx, row in df.iterrows():
                    val = row.get(name)
                    try:
                        f_val = float(val)
                        per_q_scores[idx][name] = f_val
                        import math
                        if not math.isnan(f_val):
                            col_vals.append(f_val)
                    except (TypeError, ValueError):
                        per_q_scores[idx][name] = None
                
                aggregate[name] = sum(col_vals) / len(col_vals) if col_vals else None
                score_str = f"{aggregate[name]:.4f}" if aggregate[name] is not None else "N/A"
                print(f"      {name}: {score_str}")

    except Exception as exc:
        print(f"      Evaluation batch failed: {exc}")
        import traceback
        print(traceback.format_exc())

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
    import config.llm_config as lc
    import config.eval_config as evc
    import config.embedding_config as ec

    load_dotenv(env_path)
    api_key = os.environ.get("OPENAI_API_KEY")

    judge_model_name = (
        judge_model_override or evc.judge_model or (lc.openai_model if lc.use_openai else lc.llm_model)
    )
    judge_base_url = judge_base_url_override or (lc.openai_base_url if lc.use_openai else None)

    print(f"\n--- RAGAS Evaluation ---")
    print(f"  Test set    : {testset_path} ({len(samples)} questions)")
    print(f"  Metrics     : {', '.join(metric_names)}")
    print(f"  Judge LLM   : {judge_model_name}")
    print(f"  Configs     : {', '.join(configs.keys())}")

    judge_llm = _build_ragas_llm(
        use_openai=lc.use_openai,
        model=judge_model_name,
        base_url=judge_base_url or "http://localhost:11434/v1",
        api_key=api_key,
    )
    judge_embeddings = _build_ragas_embeddings(ec, api_key)

    report = EvalReport(
        testset_path=testset_path,
        num_questions=len(samples),
        metrics=metric_names,
    )

    run_configs = configs
    if config_filter:
        run_configs = {k: v for k, v in configs.items() if k in config_filter}

    for i, (name, overrides) in enumerate(run_configs.items()):
        print(f"\n[{i + 1}/{len(run_configs)}] Evaluating config: {name}")
        if overrides:
            print(f"  Overrides: {overrides}")

        result = _evaluate_config(
            config_name=name, overrides=overrides, samples=samples,
            metric_names=metric_names, judge_llm=judge_llm,
            judge_embeddings=judge_embeddings, env_path=env_path,
        )
        report.configs.append(result)

    return report