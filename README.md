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
    Create a `.env` file for storing your API keys and other sensitive information:
    ```bash
    os.environ["OPENAI_API_KEY"]= your api key
    ```
    Add the required keys to the `.env` file.

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
- **utils.py**: Utility functions for various tasks like loading data, handling models, etc.
- **Dockerfile**: Docker configuration for containerizing the application.
- **requirements.txt**: List of required Python packages.
- **packages.txt**: Dependencies for managing additional packages.
- **.gitignore**: Ensures sensitive files and folders are not tracked by Git.
- **.env**: Stores environment variables such as API keys.

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

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
