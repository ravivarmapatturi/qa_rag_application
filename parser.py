from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader,PDFMinerLoader
from langchain_core.documents import Document
import pypdfium2 as pdfium
from openai import OpenAI
# from markitdown import MarkItDown
from docling.document_converter import DocumentConverter






def PARSING_PDF(parsing_strategy,pdf_path):
    if parsing_strategy=="PyPDFLoader":
        loader = PyPDFLoader(pdf_path)

        langchain_docs = loader.load()
    
    elif parsing_strategy=="PyMuPDFLoader":
        loader = PyMuPDFLoader(pdf_path)

        langchain_docs = loader.load()
        
    elif parsing_strategy=="PDFMinerLoader":
        loader = PDFMinerLoader(pdf_path)
        langchain_docs = loader.load()
        
    elif parsing_strategy=="pdfium":
        # Load the PDF
        pdf = pdfium.PdfDocument(pdf_path)
        
        # List to hold LangChain documents
        langchain_docs = []
        
        # Extract metadata using get_metadata_dict()
        metadata = pdf.get_metadata_dict()
        source_metadata = {
            "source": pdf_path,
            "title": metadata.get("Title", "Unknown"),
            "author": metadata.get("Author", "Unknown"),
            "subject": metadata.get("Subject", "Unknown"),
        }
        
        # Extract data page by page
        for page_number in range(len(pdf)):
            page = pdf[page_number]
            
            # Extract text
            text = page.get_textpage().get_text_range()
            
            # Create metadata for the current page
            page_metadata = {
                "page_number": page_number + 1,
                **source_metadata,  # Add general metadata
            }
            
            # Create a LangChain Document for the current page
            document = Document(
                page_content=text,
                metadata=page_metadata
            )
            langchain_docs.append(document)
        
        return langchain_docs

    
    # elif parsing_strategy=="markitdown":
        
    #     client = OpenAI()
    #     markitdown = MarkItDown(llm_client=client, llm_model="gpt-4")
        
    #     # Convert the Markdown file
    #     result = markitdown.convert(pdf_path)
        
    #     # Access the attributes of the result object
    #     title = result.title or "Unknown"
    #     text_content = result.text_content or ""
        
    #     # Metadata
    #     metadata = {
    #         "source": pdf_path,
    #         "title": title,
    #     }
        
    #     # Create a LangChain Document
    #     langchain_docs = [
    #         Document(
    #             page_content=text_content,
    #             metadata=metadata
    #         )
    #     ]
        
    #     return langchain_docs
    
    
    elif parsing_strategy=="docling":
        
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        
        return result.document
        

        
        