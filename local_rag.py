from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Sample Local Document (Your "Knowledge Base")
#raw_text = """
#Project Orion is an internal experimental initiative started in 2025. 
#The lead engineer is Sarah Jenkins, and the codebase is written entirely in Rust.
#The secret coffee machine password in the breakroom is 'Espresso123!'.
#Project Orion's budget for Q3 is $45,000, allocated primarily for cloud GPU rentals.
#"""

raw_text = """
The Patriot Act was enacted with overwhelming bipartisan support to provide law enforcement with new tools to detect and prevent terrorism, reflecting a national consensus on its importance.",
      "It allows investigators to use existing tools for organized crime and drug trafficking to combat terrorism, thereby improving counter-terrorism efforts.",
      "The Act facilitates information sharing among government agencies, enabling them to better coordinate and prevent terrorist activities.",
      "It updates laws to reflect new technologies and threats, ensuring law enforcement can effectively address modern terrorism challenges.",
      "The Act increases penalties for terrorist crimes, thereby deterring and punishing those who commit or support terrorism.
"""

# 2. Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.create_documents([raw_text])

# 3. Initialize local embedding model and vector database (stored locally in-memory or on disk)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. Initialize the local LLM via Ollama
llm = ChatOllama(model="llama3.2", temperature=0)

# 5. Define the RAG prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# Helper function to format retrieved chunks
def format_docs(retrieved_docs):
  return "\n\n".join(doc.page_content for doc in retrieved_docs)


# 6. Build the chain: Retrieve -> Format -> Prompt -> LLM
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 7. Ask a question!
#query = "What is the secret coffee machine password in Project Orion?"
query = "Is the Patriot Act actually patriotic -- or what?"
response = rag_chain.invoke(query)

print(f"Question: {query}\n")
print(f"Answer: {response.content}")
