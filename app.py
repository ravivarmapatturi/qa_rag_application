import os
import yaml 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import pandas as pd
import tempfile
import time
import tabula
import pdfplumber
import shutil
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
import ast
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pdfminer.high_level import extract_text
from langchain_core.messages import HumanMessage, AIMessage ,SystemMessage
import re
import datetime
from query_translation import prompt,chatgpt,rag_chain,rag_chain_multi_query,generate_queries,get_unique_union,reciprocal_rank_fusion,generate_queries_decomposition,decomposition_prompt,format_qa_pair,generate_queries_step_back,response_prompt,generate_docs_for_retrieval,embeddings
from langchain.chains import create_retrieval_chain
from chunking_strategies import CHUNKING_STRATEGY
from parser import PARSING_PDF
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
# __import__('pysqlite3')  # Import the pysqlite3 module
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["GROQ_API_KEY"] = st.secrets["groq"]
# Streamlit UI setup
# st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")


st.title("Welcome to QA RAG Chatbot")
placeholder = st.empty()





# File uploader
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True
)


chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy for RAG:",
    [   
        "RecursiveCharacterTextSplitter",
        "CharacterTextSplitter",
        "titoken",
        "semantic"
    ]
    )

llm_model=st.sidebar.selectbox("Select the availabe LLM Models",
                               ["mistral",
                                "gpt-3.5-turbo"
                                ])



parsing_strategy = st.sidebar.selectbox(
    "Parsing Strategy for RAG:",
    [   
        "pdfium",
        "PyMuPDFLoader",
        "PyPDFLoader",
        "PDFMinerLoader",
        "markitdown",
        "docling"
    ]
    )


prompting_method = st.sidebar.selectbox(
    "Select the type of prompting for RAG:",
    [
        "Default (Based on User Query)",
        "Multi-Query",
        "RAG Fusion",
        "Decomposition",
        "Step Back",
        "HyDE"
    ]
    )



    


# Check if any files are uploaded
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
else:
    # Check if documents have already been processed
    if 'docs' not in st.session_state:
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
        
        
        # step 1: parsing the pdf
        docs=PARSING_PDF(parsing_strategy,temp_filepath)
        
     
    
        # Step 2: Split documents into chunks
        
        text_splitter=CHUNKING_STRATEGY(chunking_strategy)
        print(text_splitter)
        
        doc_chunks = text_splitter.split_documents(docs)

        # Step 3: Convert chunks into embeddings
        persist_directory = os.path.join(os.getcwd(), "vector_db")

        # Check if the directory exists, and create it if not
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory)
        else:
            os.makedirs(persist_directory)
            
        
                
        # embeddings = OpenAIEmbeddings()
        vector_db = Chroma.from_documents(doc_chunks, embeddings,persist_directory=persist_directory)

        # Step 4: Store documents and vector DB in session state for future use
        st.session_state.docs = docs
        st.session_state.vector_db = vector_db
        st.session_state.embeddings = embeddings
        st.session_state.text_splitter = text_splitter

        placeholder.info(f"Total documents split into {len(doc_chunks)} chunks.")
        placeholder.info(f"Stored vector DB for future use.")


    similarity_retriever = st.session_state.vector_db.as_retriever(search_type="similarity",
                                                search_kwargs={"k": 3})

    # retriever = st.session_state.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})
    
    
    
    



    

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=similarity_retriever, llm=chatgpt
    )
    






    # Input for asking questions
    chat_history=[]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        

    # Display chat history messages
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="font-size: 16px;">{message.content}</p>
                    </div>
                """, unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="font-size: 16px;">{message.content}</p>
                    </div>
                """, unsafe_allow_html=True)
        elif isinstance(message, SystemMessage):
            continue  # Skip system messages in the chat UI

    # Get user input
    question = st.chat_input("Message me :")
    if question is not None and question != "":
        st.session_state.chat_history.append(HumanMessage(content=question))
        with st.chat_message("Human"):
            st.markdown(question)

    # Process the question if provided
    if question:
        placeholder.info("Searching for context and generating the answer...")

        if prompting_method == "Default (Based on User Query)" or prompting_method is None:
            placeholder.info("Generating answer based on User Query Prompting method..")

            # Retrieve relevant documents based on the question
            context = similarity_retriever.get_relevant_documents(question)
            # Join relevant context pieces into a single string
            retrieved_data=["\n".join(doc.page_content) for doc in context]
            context_text = "\n".join([doc.page_content for doc in context])
            # Retrieve conversation history as text
            # conversation_history = [msg.content for msg in st.session_state.chat_history]
            # Combine context and conversation history
            # full_context = context_text + "\n" + conversation_history if conversation_history else context_text

            if context_text:
                # Create a history-aware retriever
                
                contextualize_q_system_prompt = """
                Given a chat history and the latest user question
                which might reference context in the chat history,
                formulate a standalone question which can be understood
                without the chat history. Do NOT answer the question,
                just reformulate it if needed and otherwise return it as is.
                """

                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                history_aware_retriever = create_history_aware_retriever(
                    chatgpt, similarity_retriever ,contextualize_q_prompt
                )

                # Define the QA prompt with the context placeholder
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an intelligent and detail-oriented QA Chatbot designed to generate comprehensive and accurate answers. 
                    Your primary task is to provide a clear, detailed, and context-aware response to the user's question.

                    Instructions:
                    - Carefully analyze the provided context and chat_history and use it to craft a complete and precise answer.
                    - If the question is ambiguous or lacks sufficient detail, take help from user .
                    - Ensure the answer is structured, easy to understand, and includes examples or explanations where necessary.
                    - use provided context whenever it is necessary ,otherwise dont use it .
                    """),
                    ("system", "Context: {context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])

                # Create the question-answering chain
                question_answer_chain = create_stuff_documents_chain(chatgpt, qa_prompt)
                # Create the retrieval-augmented generation (RAG) chain
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # Define a timestamp for display
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Invoke the chain with the required parameters, including context
                result =rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})['answer']
                


                with st.chat_message("AI"):
                    st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                            <p style="color: #777; font-size: 12px;">{timestamp}</p>
                            <p style="font-size: 16px;">{result}</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.session_state.chat_history.append(AIMessage(content=result))
                st.sidebar.info("I retrieved the data from this source:")
                st.sidebar.markdown(context_text)
            else:
                st.write("Sorry, I couldn't find any relevant context.")

        elif prompting_method=="Multi-Query": 
            placeholder.info("generating answer based on Multi-Query Prompting method..")
            # Retrieve
            # question = "What is task decomposition for LLM agents?"
            retrieval_chain = generate_queries | mq_retriever.map() | get_unique_union
            context = retrieval_chain.invoke({"question": question})
            context_text = "\n".join([doc.page_content for doc in context])
            
            if context_text:
                result = rag_chain_multi_query.run({"question": question})
                result2=rag_chain.invoke({"question":question,"context":context_text})
                # print(result2)
                with st.chat_message("AI"):
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="color: #777; font-size: 12px;">{timestamp}</p>
                        <p style="font-size: 16px;">{result2['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(result2['text']))
                st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                st.sidebar.markdown(f"questions.. ,{result}")
            else:
                st.write("Sorry, I couldn't find any relevant context.")

        elif prompting_method=="RAG Fusion":
            placeholder.info("generating answer based on Multi-Query RAG Fusion Prompting method..")
            retrieval_chain = generate_queries | mq_retriever.map() | reciprocal_rank_fusion
            context_text = retrieval_chain.invoke({"question": question})



            if context_text:
                result = rag_chain_multi_query.run({"question": question})
                result2=rag_chain.invoke({"question":question,"context":context_text})
                with st.chat_message("AI"):
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="color: #777; font-size: 12px;">{timestamp}</p>
                        <p style="font-size: 16px;">{result2['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(result2['text']))
                st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                st.sidebar.markdown(f"questions.. ,{result}")
            else:
                st.write("Sorry, I couldn't find any relevant context.")


        elif prompting_method == "Decomposition":
            questions = generate_queries_decomposition.invoke({"question":question})

            q_a_pairs = ""
            for q in questions:
                rag_chain = (
                {"context": itemgetter("question") | mq_retriever, 
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs")} 
                | decomposition_prompt
                | chatgpt
                | StrOutputParser())

                answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
                q_a_pair = format_qa_pair(q,answer)
                q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

            with st.chat_message("AI"):
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #777; font-size: 12px;">{timestamp}</p>
                    <p style="font-size: 16px;">{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(answer))
                # st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                # st.sidebar.markdown(f"questions.. ,{result}")

        elif prompting_method=="Step Back":

            # question = "What is task decomposition for LLM agents?"
            generate_queries_step_back.invoke({"question": question})

            chain = (
                {
                    # Retrieve context using the normal question
                    "normal_context": RunnableLambda(lambda x: x["question"]) | mq_retriever,
                    # Retrieve context using the step-back question
                    "step_back_context": generate_queries_step_back | mq_retriever,
                    # Pass on the question
                    "question": lambda x: x["question"],
                }
                | response_prompt
                | chatgpt
                | StrOutputParser()
            )

            answer=chain.invoke({"question": question})

            with st.chat_message("AI"):
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #777; font-size: 12px;">{timestamp}</p>
                    <p style="font-size: 16px;">{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(answer))
                # st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                # st.sidebar.markdown(f"questions.. ,{result}")

        elif prompting_method == "HyDE":
            

            # Run
            # question = "What is task decomposition for LLM agents?"
            generate_docs_for_retrieval.invoke({"question":question})


            retrieval_chain = generate_docs_for_retrieval | mq_retriever 
            retireved_docs = retrieval_chain.invoke({"question":question})

            # final_rag_chain = (
            #     prompt
            #     | chatgpt
            #     | StrOutputParser()
            # )

            answer=rag_chain.invoke({"context":retireved_docs,"question":question})

            with st.chat_message("AI"):
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #777; font-size: 12px;">{timestamp}</p>
                    <p style="font-size: 16px;">{answer['text']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(answer['text']))
                # st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                # st.sidebar.markdown(f"questions.. ,{result}")
                
        else:
            # Default (Based on User Query)
            pass