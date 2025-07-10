from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool

# Load your vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "vectorstore_openai",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

# Create QA chain using your vector store
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


@tool
def qa_from_pdf(query: str) -> str:
    """
    Answers using only PDF content.
    Returns a fallback message if no relevant data is found.
    """
    result = qa_chain.invoke({"query": query})
    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

    if not source_docs or len(answer.strip()) < 20:
        return (
            "Sorry, the document does not contain relevant "
            "information about that topic."
        )

    return answer
