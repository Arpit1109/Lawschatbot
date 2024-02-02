import textwrap
import numpy as np
import pandas as pd
# import faiss
import openai
import langchain
# import PyPDF2
from langchain.llms import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

# pdf_path = os.path.join("data", "lawsofpower.pdf")

openai.api_key=OPEN_API_KEY



# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embed_model = "text-embedding-ada-002"
llm="gpt-3.5-turbo"

# embeddings = []
# paragraph = []

# def extract_text_from_pdf(pdf_path):
#     text = []
#     with open(pdf_path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)
#         num_pages = len(reader.pages)
#         print("NUM Pages:",num_pages)
#         for page in range(num_pages):
#             pdf_page = reader.pages[page]
#             text.append(pdf_page.extract_text())
#     return text



# def embeddings_creation(data):
#         chunks=data
#         # List to store the embeddings
#         # Generate embeddings for each chunk
#         for chunk in chunks:
#             paragraph.append(chunk)
#             response = embedding.embed_query(chunk)
#             embeddings.append(response)
#         return embeddings, paragraph

# def embeddings_to_indexer(embeddings):
#         nlist = 1
#         # Convert the embeddings to a numpy array
#         embeddings = np.array(embeddings)
#         # Set the dimension of the embeddings
#         embedding_dim = embeddings.shape[1]
#         print("Embedding Dimension:",embedding_dim)
#         # print("Embedding Dimension:",embedding_dim)
#         # Initialize the index
#         # Add the embeddings to the index
#         quantiser = faiss.IndexFlatL2(embedding_dim)
#         indexer = faiss.IndexIVFFlat(quantiser, embedding_dim, nlist,   faiss.METRIC_L2)
        
#         #train the model
#         indexer.train(embeddings)
#         indexer.add(embeddings)
#         return indexer

def question_query(question):
        res = openai.Embedding.create(
                   input=question,
                  engine=embed_model
                 )
        query_vector =res['data'][0]['embedding']
        return query_vector

from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("canopy--document-uploader")

def question_answering_model(query_vector):
    res = index.query(vector=query_vector, top_k=2, include_metadata=True)
    text= [match['metadata']['text'] for match in res['matches']]
    relevant_documents = ''

    for i in text:
        # Replace this with your actual code to retrieve the relevant documents based on the index
        relevant_documents+=i

    # print(relevant_documents)
    return relevant_documents

def make_prompt(user_query, relevant_documents):
  escaped = relevant_documents.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  QUESTION: '{user_query}'
  PASSAGE: '{relevant_documents}'
  """).format(user_query=user_query, relevant_documents=escaped)

  return prompt


def get_completion(prompt, model=llm):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def predict(question):
    #  embeddings, paragraph = embeddings_creation(data)
     
     query_vector = question_query(question)
     relevant_documents = question_answering_model(query_vector)
     final=make_prompt(question,relevant_documents)
     answer=get_completion(final)
     return answer
