"""Run the fixed RAGAS eval suite against the default RAG pipeline.

Builds a fresh index from the fixture PDFs in eval/fixtures/docs/, answers
every question in eval/testset.json through the same hybrid (dense+BM25)
retriever + generation chain app.py uses for its default prompting method,
and scores the results with RAGAS (faithfulness, answer relevancy, context
precision, context recall) -- see rag_pipeline.py for the pipeline itself;
this script is just the harness plus the RAGAS wiring, no scoring logic of
our own.

Requires OPENAI_API_KEY and/or GROQ_API_KEY (whichever `query_translation.py`
is configured to use) in the environment or a .env file -- both the RAG
pipeline's generation step and RAGAS's own LLM-graded metrics need a real
model to call.

Usage:
    python eval/run_eval.py [--threshold 0.7] [--report eval/results/latest.json]
    python eval/run_eval.py --contextual --rerank --report eval/results/contextual_reranked.json
    python eval/run_eval.py --agentic --report eval/results/agentic.json

--contextual and --rerank toggle contextual-retrieval chunking
(contextual_retrieval.py) and cross-encoder reranking (rag_pipeline.
build_reranking_retriever) on top of the baseline hybrid retriever.
--agentic routes every question through the LangGraph router
(agentic_rag.py: simple/multi_hop/complex) instead of the plain hybrid
pipeline -- mutually exclusive with --contextual/--rerank, since the router
builds its own index and GraphRAG-style multi-hop graph internally. Every
mode writes to its own --report file so a naive-vs-improved-vs-agentic run
can be compared directly.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DEFAULT_FIXTURES_GLOB = os.path.join(HERE, "fixtures", "docs", "*.pdf")
DEFAULT_TESTSET_PATH = os.path.join(HERE, "testset.json")
DEFAULT_REPORT_PATH = os.path.join(HERE, "results", "latest.json")

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_dataset(
    testset_path: str,
    fixtures_glob: str,
    use_contextual: bool = False,
    use_rerank: bool = False,
    use_agentic: bool = False,
):
    """Run every fixed-testset question through the pipeline; return a RAGAS
    EvaluationDataset plus the raw per-question records (for the report)."""
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    pdf_paths = sorted(glob.glob(fixtures_glob))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No fixture PDFs found at {fixtures_glob} -- run "
            "eval/fixtures/generate_fixtures.py first."
        )

    from cost_tracking import timed, total_tokens_and_cost
    from observability import usage_tracking_config

    if use_agentic:
        from agentic_rag import build_app_from_pdfs

        app = build_app_from_pdfs(pdf_paths)

        def ask(question: str) -> dict:
            config, usage_handler = usage_tracking_config(run_name="eval agentic")
            with timed() as t:
                result = app.invoke({"question": question}, config=config)
            contexts = [doc.page_content for doc in result.get("documents", [])]
            return {
                "answer": result["generation"],
                "contexts": contexts,
                "latency_seconds": t.elapsed_seconds,
                "usage": total_tokens_and_cost(usage_handler.usage_metadata),
            }

    else:
        from rag_pipeline import answer, build_hybrid_retriever, build_qa_chain, build_reranking_retriever, build_vector_store, load_and_chunk

        chunks = load_and_chunk(pdf_paths, use_contextual=use_contextual)
        vector_store = build_vector_store(chunks)
        # When reranking, retrieve a wider candidate pool (k=10) for the
        # reranker to actually choose among; otherwise the standard k=3.
        retriever = build_hybrid_retriever(vector_store, chunks, k=10 if use_rerank else 3)
        if use_rerank:
            retriever = build_reranking_retriever(retriever, top_n=3)
        chain = build_qa_chain(retriever)

        def ask(question: str) -> dict:
            _, usage_handler = usage_tracking_config()
            with timed() as t:
                result = answer(question, chain, usage_handler=usage_handler)
            result["latency_seconds"] = t.elapsed_seconds
            result["usage"] = total_tokens_and_cost(usage_handler.usage_metadata)
            return result

    records = []
    samples = []
    latencies = []
    for item in testset:
        result = ask(item["question"])
        latencies.append(result["latency_seconds"])
        record = {
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "latency_seconds": result["latency_seconds"],
            "usage": result["usage"],
        }
        records.append(record)
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=item["ground_truth"],
            )
        )

    return EvaluationDataset(samples=samples), records, latencies


def score(dataset) -> dict:
    """Score with RAGAS's own metric implementations -- no custom scoring here."""
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    from query_translation import chatgpt, embeddings

    ragas_llm = LangchainLLMWrapper(chatgpt)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas()[METRIC_NAMES].mean().to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum acceptable mean score per metric.")
    parser.add_argument("--testset", default=DEFAULT_TESTSET_PATH)
    parser.add_argument("--fixtures-glob", default=DEFAULT_FIXTURES_GLOB)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--contextual", action="store_true", help="Use contextual-retrieval chunking.")
    parser.add_argument("--rerank", action="store_true", help="Add cross-encoder reranking on top of hybrid retrieval.")
    parser.add_argument("--agentic", action="store_true", help="Route through the LangGraph agentic router instead of the plain pipeline.")
    args = parser.parse_args()

    dataset, records, latencies = build_dataset(
        args.testset,
        args.fixtures_glob,
        use_contextual=args.contextual,
        use_rerank=args.rerank,
        use_agentic=args.agentic,
    )
    scores = score(dataset)

    from cost_tracking import latency_summary

    total_cost_usd = sum(r["usage"]["cost_usd"] for r in records if r["usage"]["cost_usd"] is not None) or None
    cost_latency = {
        "latency": latency_summary(latencies),
        "total_input_tokens": sum(r["usage"]["input_tokens"] for r in records),
        "total_output_tokens": sum(r["usage"]["output_tokens"] for r in records),
        "total_cost_usd": total_cost_usd,
        "cost_usd_per_query": (total_cost_usd / len(records)) if total_cost_usd is not None and records else None,
    }

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "cost_latency": cost_latency, "records": records}, f, indent=2)

    print("\nRAGAS eval results (mean over %d questions):" % len(records))
    failed = []
    for name in METRIC_NAMES:
        value = scores.get(name)
        status = "OK" if value is not None and value >= args.threshold else "BELOW THRESHOLD"
        if status != "OK":
            failed.append(name)
        print(f"  {name:<20} {value:.3f}  [{status}]" if value is not None else f"  {name:<20} n/a")

    lat = cost_latency["latency"]
    print("\nLatency: p50=%.2fs p95=%.2fs mean=%.2fs" % (lat["p50_seconds"], lat["p95_seconds"], lat["mean_seconds"]))
    if cost_latency["total_cost_usd"] is not None:
        print("Cost: $%.5f total, $%.5f/query (%d input + %d output tokens)" % (
            cost_latency["total_cost_usd"], cost_latency["cost_usd_per_query"],
            cost_latency["total_input_tokens"], cost_latency["total_output_tokens"],
        ))
    else:
        print(f"Cost: unknown model pricing -- tokens used: {cost_latency['total_input_tokens']} in / {cost_latency['total_output_tokens']} out (see cost_tracking.MODEL_PRICING_PER_MILLION_TOKENS)")
    print(f"\nFull report written to {args.report}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)} below threshold {args.threshold}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
