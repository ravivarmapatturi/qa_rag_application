from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings




def CHUNKING_STRATEGY(chunking_strategy):
    if chunking_strategy=="CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(chunk_size = 1024, chunk_overlap=200, separator='', strip_whitespace=False)
    elif chunking_strategy=="RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, chunk_overlap=200, length_function=len,
                separators = [
                    "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",],
                is_separator_regex=False)
        
    elif chunking_strategy=="tiktoken":
        print("tiktoken")
        text_splitter = TokenTextSplitter(model_name='gpt-3.5-turbo',
                                  chunk_size=1024,
                                  chunk_overlap=200)
        
    elif chunking_strategy=="semantic":
        print("semantic")
        # Percentile - all differences between sentences are calculated, and then any difference greater than the X percentile is split
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        text_splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile"
    )
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
    return text_splitter