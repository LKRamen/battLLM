"""
chatbot.py

Usage: streamlit run chatbot.py

To utilize the url, apikey, and project_id fields in the WatsonxLLM function call, you must create your own IBM Cloud account.
Rose-Hulman students have access to free student plans where they can access such resources. You must then create a project
with Watson and utilize the given API Key and Project ID.

For any other questions email me: lkrahman08@gmail.com
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ibm import WatsonxLLM

load_dotenv()

watsonx_llm = WatsonxLLM(
    model_id='meta-llama/llama-3-3-70b-instruct',
    url=os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com'),
    apikey=os.getenv('WATSONX_API_KEY'),
    project_id=os.getenv('WATSONX_PROJECT_ID'),
    params={
        'decoding_method': 'greedy',  # More deterministic
        'max_new_tokens': 500,        # More tokens for complete answers
        'temperature': 0.1,           # Lower temperature for more focused responses
        'repetition_penalty': 1.1,    # Reduce repetition
    },
)

st.title('Ask LLM(Layth Learning Model)')

@st.cache_resource
def load_pdf():
# Update PDF name here to whatever you like
 #pdfs = ['./BattChallengeReqs/BMS Software Requirement.pdf', './BattChallengeReqs/BWC_-_Software_Requirements_Document_-_Rev5.pdf']
 documents = []
 pdf_folder_path = 'BattChallengeReqs'
 for file in os.listdir(pdf_folder_path):
  if file.endswith('.pdf'):
   pdf_path = os.path.join(pdf_folder_path, file)
   loader = PyPDFLoader(pdf_path)
   documents.extend(loader.load())
   embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

   # Split documents into chunks
   text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250
   )
   splits = text_splitter.split_documents(documents)

   vector_store = FAISS.from_documents(splits, embeddings)

  return vector_store

vector_store = load_pdf()

if 'messages' not in st.session_state:
 st.session_state.messages = []

for message in st.session_state.messages:
 st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass Your Prompt here')

if prompt:
 # Display the prompt
 st.chat_message('user').markdown(prompt)
#Store the user prompt in state
 st.session_state.messages.append({'role':'user', 'content':prompt})
 # Create the prompt template

 # Create the prompt template
 system_prompt = (
  "You are a Battery Management System (BMS) technical expert. "
  "You have deep knowledge of battery control systems, safety protocols, "
  "fault detection, power management, and automotive battery systems. "
  "Answer questions based on the provided technical documentation and your own technical expertise . "
  "Use precise technical terminology and explain complex concepts clearly. "
  "If discussing safety-critical systems, emphasize safety considerations."
  "\n\nTechnical Context:\n{context}\n\nQuestion: {input}\n\nExpert Analysis:"
 )

 prompt_template = ChatPromptTemplate.from_messages([
  ("human", system_prompt),
 ])
# Create documents chain
 question_answer_chain = create_stuff_documents_chain(watsonx_llm, prompt_template)
# Create rag chain
 rag_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)
# Send the prompt to the Q&A chain
 response = rag_chain.invoke({"input": prompt})
 #Debug
 with st.expander("Debug Information"):
  st.write("**Retrieved Documents:**")
  for i, doc in enumerate(response.get("context", [])):
   st.write(f"**Document {i + 1}:**")
   st.write(f"Page {doc.metadata['page']} of {doc.metadata['title']}")
   st.write(doc.page_content[:200] + "...")
   st.write("---")
# Show the llm response
 st.chat_message('assistant').markdown(response['answer'])
 #Store the llm response in state
 st.session_state.messages.append({
  'role': 'assistant', 'content': response['answer']
