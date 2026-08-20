# QA RAG Chat Application

This repository contains a QA RAG (Retrieval-Augmented Generation) chat application that allows for dynamic querying and information retrieval from various document formats. The app provides flexibility in how the text is chunked, parsed, and prompted, offering a highly customizable experience for text extraction and question answering tasks.

## Features

- **Chunking Strategies**: Choose from different text chunking methods to suit your needs for document processing:
    - `RecursiveCharacterTextSplitter`
    - `CharacterTextSplitter`
    - `Titoken`
    - `Semantic`

- **Parsing Strategies**: Select the most suitable strategy for parsing various document types:
    - `pdfium`
    - `PyMuPDFLoader`
    - `PyPDFLoader`
    - `PDFMinerLoader`

- **Prompting Methods**: Customize your prompting approach to control the flow and depth of the answers:
    - `Default (Based on User Query)`
    - `Multi-Query`
    - `RAG Fusion`
    - `Decomposition`
    - `Step Back`
    - `HyDE`

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
- **rag_pipeline.py**: The single-turn parse → chunk → index → hybrid-retrieve → generate pipeline as plain importable functions, shared by the eval harness (app.py's interactive chat has its own history-aware variant of the same pipeline).
- **eval/**: RAGAS-based eval suite -- fixed test set, fixture documents, and the eval runner. See [Evaluation](#evaluation) below.
- **Dockerfile**: Docker configuration for containerizing the application.
- **requirements.txt**: List of required Python packages.
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

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
