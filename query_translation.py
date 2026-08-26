import os
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Optional
from pydantic import BaseModel, PrivateAttr
from langchain_core.runnables import RunnableSerializable
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load OPENAI_API_KEY / GROQ_API_KEY from a local .env file if present (see
# README setup instructions). Falls back to whatever's already in the real
# environment (CI secrets, `export`, Streamlit Cloud secrets synced to env,
# etc.) -- this module needs to be importable outside a Streamlit runtime
# (eval scripts, CI) so it does not depend on `st.secrets`.
load_dotenv()


# chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)




model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


def get_llm(model_name: str = "mistral"):
    """Build a chat model for the given sidebar selection.

    Every chain in this module is built via the `build_*` functions below,
    each taking an `llm` produced by this factory -- so app.py's model
    dropdown actually changes which model answers, instead of every
    prompting method being permanently wired to whatever model was live at
    import time.
    """
    if model_name in ("gpt-3.5-turbo", "gpt-4"):
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    if model_name == "mistral":
        return ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    raise ValueError(f"Unknown llm_model: {model_name}")


# Fixed default used by callers that don't care which model answers (the
# eval harness in particular wants a stable, reproducible model across runs
# rather than tracking whatever a Streamlit session's sidebar happens to be
# set to) -- rag_pipeline.py and eval/run_eval.py both rely on this default.
chatgpt = get_llm("mistral")












# with open("/home/ravivarma/Downloads/preplaced/session_5_tasks/miniproject/credentials/api_keys.yaml") as file:
#     config = yaml.safe_load(file)
# api_keys = config['api_keys']["chatgpt"]
# os.environ["OPENAI_API_KEY"] = api_keys

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)


qa_template1 = """
You are an intelligent and detail-oriented QA Chatbot designed to generate comprehensive and accurate answers. 
Your primary task is to provide a clear, detailed, and context-aware response to the user's question.

Instructions:
- Carefully analyze the provided context and use it to craft a complete and precise answer.
- If the question is ambiguous or lacks sufficient detail, take help from user .
- Ensure the answer is structured, easy to understand, and includes examples or explanations where necessary.
=======
- use provided context whenever it is necessary ,otherwise dont use it .

Context:
{context}

Question: {question}

Answer:
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template1)


def build_rag_chain(llm):
    return LLMChain(prompt=qa_prompt, llm=llm)


rag_chain = build_rag_chain(chatgpt)



#########################################################################################
### multi-query
##########################################################################################




qa_template2 = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""



prompt_perspectives = ChatPromptTemplate.from_template(qa_template2)


def build_generate_queries(llm):
    return prompt_perspectives | llm | StrOutputParser() | (lambda x: x.split("\n"))


generate_queries = build_generate_queries(chatgpt)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]





def build_rag_chain_multi_query(llm):
    return LLMChain(prompt=prompt_perspectives, llm=llm)


rag_chain_multi_query = build_rag_chain_multi_query(chatgpt)


#########################################################################################


def reciprocal_rank_fusion(results: list[list], k=60):
                """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
                    and an optional parameter k used in the RRF formula """
                
                # Initialize a dictionary to hold fused scores for each unique document
                fused_scores = {}

                # Iterate through each list of ranked documents
                for docs in results:
                    # Iterate through each document in the list, with its rank (position in the list)
                    for rank, doc in enumerate(docs):
                        # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                        doc_str = dumps(doc)
                        # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                        if doc_str not in fused_scores:
                            fused_scores[doc_str] = 0
                        # Retrieve the current score of the document, if any
                        previous_score = fused_scores[doc_str]
                        # Update the score of the document using the RRF formula: 1 / (rank + k)
                        fused_scores[doc_str] += 1 / (rank + k)

                # Sort the documents based on their fused scores in descending order to get the final reranked results
                reranked_results = [
                    (loads(doc), score)
                    for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                ]

                # Return the reranked results as a list of tuples, each containing the document and its fused score
                return reranked_results
            
################################# decomposition ################################################################################3

template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)


def build_generate_queries_decomposition(llm):
    return prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))


generate_queries_decomposition = build_generate_queries_decomposition(chatgpt)


template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)


def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()




##################################################step back ##############33



# Few Shot Examples

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
step_back_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)


def build_generate_queries_step_back(llm):
    return step_back_prompt | llm | StrOutputParser()


generate_queries_step_back = build_generate_queries_step_back(chatgpt)




response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)



############################################################## hyde


template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)


def build_generate_docs_for_retrieval(llm):
    return prompt_hyde | llm | StrOutputParser()


generate_docs_for_retrieval = build_generate_docs_for_retrieval(chatgpt)
