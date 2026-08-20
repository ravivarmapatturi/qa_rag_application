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


def build_dataset(testset_path: str, fixtures_glob: str):
    """Run every fixed-testset question through the pipeline; return a RAGAS
    EvaluationDataset plus the raw per-question records (for the report)."""
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    from rag_pipeline import answer, build_hybrid_retriever, build_qa_chain, build_vector_store, load_and_chunk

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    pdf_paths = sorted(glob.glob(fixtures_glob))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No fixture PDFs found at {fixtures_glob} -- run "
            "eval/fixtures/generate_fixtures.py first."
        )

    chunks = load_and_chunk(pdf_paths)
    vector_store = build_vector_store(chunks)
    retriever = build_hybrid_retriever(vector_store, chunks)
    chain = build_qa_chain(retriever)

    records = []
    samples = []
    for item in testset:
        result = answer(item["question"], chain)
        record = {
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": result["answer"],
            "contexts": result["contexts"],
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

    return EvaluationDataset(samples=samples), records


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
    args = parser.parse_args()

    dataset, records = build_dataset(args.testset, args.fixtures_glob)
    scores = score(dataset)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "records": records}, f, indent=2)

    print("\nRAGAS eval results (mean over %d questions):" % len(records))
    failed = []
    for name in METRIC_NAMES:
        value = scores.get(name)
        status = "OK" if value is not None and value >= args.threshold else "BELOW THRESHOLD"
        if status != "OK":
            failed.append(name)
        print(f"  {name:<20} {value:.3f}  [{status}]" if value is not None else f"  {name:<20} n/a")
    print(f"\nFull report written to {args.report}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)} below threshold {args.threshold}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
