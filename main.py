
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
load_dotenv()



pdf_path="docs/plaq.pdf"

loader=PyPDFLoader(pdf_path)
documents=loader.load()
print(documents) # On a une liste de documents représentants chacune des page via l'objet Document

text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator='\n')
docs=text_splitter.split_documents(documents=documents)
print(len(docs))

embeddings=OpenAIEmbeddings()
vectorsstore=FAISS.from_documents(docs,embeddings)
vectorsstore.save_local("faiss_index")
new_vectorstore=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

qa=RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=new_vectorstore.as_retriever()
)


query=input("Votre question : ")
result=qa({"query":query})
print(result)