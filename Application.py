import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# STEP 1: Load .env and API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# STEP 2: Load PDF
pdf_path = (
    r"C:\Users\ITSolutions\Desktop\Rag_Langchain task\\"
    r"primary_healthcare_quality.pdf"
)
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# STEP 3: Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# STEP 4: Create vector store with OpenAI embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("vectorstore_openai")

# STEP 5: Reload vector store + create retriever
vectorstore = FAISS.load_local(
    "vectorstore_openai",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# STEP 6: Use OpenAI Chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# STEP 7: Build QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# STEP 8: Ask question
query = "what is primary health care according to WHO?"
response = qa_chain.run(query)

print("\n🔍 Answer:", response)
