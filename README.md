# QA RAG Chat Application

This repository contains a QA RAG (Retrieval-Augmented Generation) chat application that allows for dynamic querying and information retrieval from various document formats. The app provides flexibility in how the text is chunked, parsed, and prompted, offering a highly customizable experience for text extraction and question answering tasks.

## Features

- **Chunking Strategies**: Choose from different text chunking methods to suit your needs for document processing:
    - `RecursiveCharacterTextSplitter`
    - `CharacterTextSplitter`
    - `tiktoken`
    - `semantic`
    - Optionally layered on top: **contextual-retrieval chunking** (an LLM-generated blurb situating each chunk within its source document, prepended before embedding -- see [Retrieval Enhancements](#retrieval-enhancements)).

- **Parsing Strategies**: Select the most suitable strategy for parsing various document types:
    - `pdfium`
    - `PyMuPDFLoader`
    - `PyPDFLoader`
    - `PDFMinerLoader`
    - `docling`

- **Hybrid retrieval**: dense (Chroma) + sparse (BM25) retrieval combined via an `EnsembleRetriever`, optionally reranked with a cross-encoder (see [Retrieval Enhancements](#retrieval-enhancements)).

- **Prompting Methods**: Customize your prompting approach to control the flow and depth of the answers:
    - `Default (Based on User Query)`
    - `Multi-Query`
    - `RAG Fusion`
    - `Decomposition`
    - `Step Back`
    - `HyDE`
    - `Agentic Router` -- classifies each question and routes it to naive retrieval, GraphRAG-style multi-hop retrieval, or a self-correcting agent loop (see [Agentic Routing](#agentic-routing--graphrag-style-multi-hop-retrieval)).

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/qa-rag-chat-application.git
    cd qa-rag-chat-application
    ```

2. **Install Requirements**:
    Ensure that Python is installed, then install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment**:
    Copy `.env.example` to `.env` and fill in your keys:
    ```bash
    cp .env.example .env
    # then edit .env:
    OPENAI_API_KEY=your-openai-key
    GROQ_API_KEY=your-groq-key
    ```
    `.env` is loaded automatically at startup via `python-dotenv`.

4. **Run the Application**:
    To start the application, run:
    ```bash
    streamlit run app.py
    ```

## File Structure

- **app.py**: The main application file that ties everything together.
- **chunking_strategies.py**: Contains different chunking strategies used for processing the text.
- **query_translation.py**: Handles query re-writing and transformation.
- **parser.py**: Includes different parsers for processing documents.
- **rag_pipeline.py**: The single-turn parse → chunk → index → hybrid-retrieve → generate pipeline as plain importable functions, shared by the eval harness (app.py's interactive chat has its own history-aware variant of the same pipeline). Also has the contextual-retrieval and cross-encoder-reranking retriever builders.
- **contextual_retrieval.py**: LLM-generated per-chunk context, following Anthropic's published contextual-retrieval technique.
- **graph_retrieval.py**: GraphRAG-style multi-hop retrieval (LLM-extracted entity/relationship graph + NetworkX traversal) -- not Microsoft's `graphrag` package, see the module docstring for why.
- **agentic_rag.py**: LangGraph router that classifies each question and sends it to naive retrieval, graph retrieval, or a CRAG-style self-correcting agent loop.
- **observability.py**: Langfuse tracing via LangChain's callback integration, shared by every prompting method and the eval harness. See [Observability](#observability).
- **eval/**: RAGAS-based eval suite -- fixed test set, fixture documents, and the eval runner. See [Evaluation](#evaluation) below.
- **tests/**: pytest unit tests, no real API keys required. See [Tests](#tests) below.
- **Dockerfile**: Docker configuration for containerizing the application.
- **requirements.txt**: List of required Python packages for running the app.
- **requirements-dev.txt**: Extra packages needed only for running the test suite (pytest).
- **packages.txt**: Dependencies for managing additional packages.
- **.gitignore**: Ensures sensitive files and folders are not tracked by Git.
- **.env.example**: Template for the `.env` file storing environment variables such as API keys.

## Usage

1. **Select Chunking Strategy**: In the sidebar, choose the chunking strategy that fits your needs. The available options are:
    - RecursiveCharacterTextSplitter
    - CharacterTextSplitter
    - Titoken
    - Semantic

2. **Select Parsing Strategy**: Choose how the documents should be parsed from the available options:
    - pdfium
    - PyMuPDFLoader
    - PyPDFLoader
    - PDFMinerLoader

3. **Select Prompting Method**: Choose the desired prompting method for the chat application:
    - Default (Based on User Query)
    - Multi-Query
    - RAG Fusion
    - Decomposition
    - Step Back
    - HyDE
    - Agentic Router

4. **Optional retrieval enhancements** (sidebar checkboxes): contextual retrieval and cross-encoder reranking, independent of which prompting method is selected. See [Retrieval Enhancements](#retrieval-enhancements).

## Retrieval Enhancements

- **Contextual retrieval** (`contextual_retrieval.py`): prepends a short LLM-generated summary situating each chunk within its source document before it's embedded and BM25-indexed, following [Anthropic's published contextual-retrieval technique](https://www.anthropic.com/news/contextual-retrieval) (same prompt template and prepend-before-embed pattern as their reference implementation). Adapted to call through this repo's existing LLM (Groq) rather than the Anthropic API directly, to avoid a third required API key for one step. One extra LLM call per chunk at indexing time -- opt-in via the sidebar checkbox.
- **Cross-encoder reranking**: retrieves a wider candidate pool from the hybrid retriever and reranks it with `cross-encoder/ms-marco-MiniLM-L-6-v2` via LangChain's `CrossEncoderReranker`/`HuggingFaceCrossEncoder` (no custom scoring logic).

## Agentic Routing / GraphRAG-Style Multi-Hop Retrieval

`agentic_rag.py` adapts LangGraph's published Adaptive-RAG / Corrective-RAG (CRAG) reference pattern: a
structured-output router classifies each question, then routes it down one of three paths:

- **simple** -- straight to the existing hybrid retriever, no extra overhead.
- **multi_hop** -- `graph_retrieval.py`'s **GraphRAG-style multi-hop retrieval**: an LLM extracts
  (entity, relation, entity) triples from every chunk into a NetworkX graph at index time, and a question is
  answered by matching entities it mentions and traversing up to N hops to pull in connected chunks. **This is
  not Microsoft's `graphrag` package** -- that library ships its own indexing pipeline (Leiden community
  detection, hierarchical community summarization) built for large corpora with real community structure, and
  wouldn't be meaningfully exercisable at this project's scale. This module implements the same conceptual
  technique (entity/relationship extraction, graph traversal) directly on the existing LangChain/NetworkX stack.
  Falls back to hybrid retrieval if no graph entities match the question.
- **complex** -- an agent loop: the question is decomposed into 2-3 sub-questions, each answered through a
  bounded CRAG-style self-correction loop (retrieve → grade each document's relevance with an LLM call → if
  none are relevant, rewrite the query and retry once → answer from the relevant documents), then the
  sub-answers are synthesized into one final answer.

## Observability

`observability.py` wires in [Langfuse](https://langfuse.com) tracing via LangChain's own callback integration
(`langfuse.langchain.CallbackHandler`) -- no hand-rolled span/timing bookkeeping. Set `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` in `.env` (free tier at [cloud.langfuse.com](https://cloud.langfuse.com), or self-host and
also set `LANGFUSE_HOST`) and every prompting method -- Default through Agentic Router, plus the eval harness --
traces automatically. Leave them unset and the app runs exactly the same with tracing silently disabled; a bad
or unreachable Langfuse endpoint fails the same way (a background warning, not a crash) so tracing can never
take the app down.

For the Agentic Router specifically, every node function receives the run's `config` from LangGraph and
forwards it into its own nested LLM calls (routing classification, per-document grading, query rewriting,
decomposition, generation), so a single question shows up as one trace in Langfuse with each routing/retrieval/
grading/generation step as a nested span -- not disconnected fragments.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Unit tests for `parser.py` (each lightweight parsing strategy against the eval fixture PDFs), `chunking_strategies.py`,
`graph_retrieval.py`'s traversal logic (hop reachability, undirected traversal, entity matching, no-match fallback),
`rag_pipeline.py`'s hybrid retrieval and answer shape, `rag_pipeline.build_reranking_retriever`, `agentic_rag.py`'s
routing/grading/decomposition logic and graph topology, and `observability.py`'s tracing-enabled/disabled behavior.
None of these need real API keys -- LLM calls that need real output use a `FakeListChatModel`; calls using structured
output (routing, grading, decomposition) use small stub LLMs that return canned Pydantic results, since
`FakeListChatModel` doesn't emulate tool-calling realistically. Runs in CI on every push via `.github/workflows/tests.yml`.

## Evaluation

Retrieval and generation quality are tracked with a small [RAGAS](https://github.com/explodinggradients/ragas) suite
against a fixed set of questions over two synthetic fixture documents (`eval/fixtures/docs/`), so changes to
chunking, retrieval, or prompting can be measured against a baseline instead of eyeballed.

```bash
pip install -r eval/requirements-eval.txt
python eval/run_eval.py
```

This builds a fresh index from the fixture PDFs, answers every question in `eval/testset.json` through the
same hybrid (dense + BM25) retriever used by the app's default prompting mode, and scores the results on
faithfulness, answer relevancy, context precision, and context recall. Results are written to
`eval/results/latest.json` and the run fails (non-zero exit) if any metric falls below `--threshold` (default
0.7) -- this is what `.github/workflows/eval.yml` runs in CI on every push/PR, given `OPENAI_API_KEY` /
`GROQ_API_KEY` repo secrets.

To edit the fixture documents, edit the `.txt` files in `eval/fixtures/sources/` and regenerate the PDFs with
`python eval/fixtures/generate_fixtures.py`.

Pass `--contextual` and/or `--rerank` to score the retrieval enhancements against the same fixed test set, or
`--agentic` to score the LangGraph router instead of the plain pipeline (mutually exclusive with the other two
-- the router builds its own index and graph internally). Each takes its own `--report` path so runs can be
compared directly:

```bash
python eval/run_eval.py --report eval/results/baseline.json
python eval/run_eval.py --contextual --rerank --report eval/results/enhanced.json
python eval/run_eval.py --agentic --report eval/results/agentic.json
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
