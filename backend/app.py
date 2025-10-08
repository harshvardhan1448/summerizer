from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuration ---
NEWS_DATA_PATH = Path("data/news")
LEGAL_DATA_PATH = Path("data/legal")
VECTORSTORE_DIR = Path("vectorstores")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

GROQ_MODEL_NAME = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set.")
if not LANGCHAIN_API_KEY:
    raise EnvironmentError("LANGCHAIN_API_KEY not set. Get one from https://smith.langchain.com/")

# --- Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain import hub
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor

# --- Embeddings & Vector Store Utilities ---
_EMBEDDINGS_SINGLETON = None
def get_embeddings():
    global _EMBEDDINGS_SINGLETON
    if _EMBEDDINGS_SINGLETON is None:
        _EMBEDDINGS_SINGLETON = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS_SINGLETON

def _load_and_chunk(path: Path, subdir: str = "") -> List:
    target_path = path / subdir if subdir else path
    if not target_path.exists(): return []
    loader = DirectoryLoader(str(target_path), glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(docs)

def _vectorstore_paths(store_name: str) -> Tuple[Path, Path]:
    base = VECTORSTORE_DIR / store_name
    return base.with_suffix(".faiss"), base.with_suffix(".pkl")

def build_or_load_vectorstore(domain: str, data_path: Path, subdir: str = "") -> FAISS:
    store_name = f"{domain}_{subdir}" if subdir else domain
    faiss_path, pkl_path = _vectorstore_paths(store_name)
    if faiss_path.exists() and pkl_path.exists():
        print(f"[INFO] Vector store for '{store_name}' found. Loading from disk...")
        try:
            return FAISS.load_local(str(VECTORSTORE_DIR), index_name=store_name, embeddings=get_embeddings(), allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[WARN] Failed to load '{store_name}': {e}. Rebuilding...")
    print(f"[INFO] Building vector store for '{store_name}' from {data_path / subdir} ...")
    docs = _load_and_chunk(data_path, subdir)
    if not docs: raise FileNotFoundError(f"No documents in {data_path / subdir}.")
    store = FAISS.from_documents(docs, get_embeddings())
    store.save_local(str(VECTORSTORE_DIR), index_name=store_name)
    print(f"[INFO] Vector store for '{store_name}' built with {len(docs)} chunks.")
    return store

# --- LLM & Vector Store Initialization ---
def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_NAME, temperature=0.2)

print("[INIT] Loading / building vector stores...")
_NEWS_ARTICLES_STORE = build_or_load_vectorstore("news", NEWS_DATA_PATH, "articles")
_NEWS_SUMMARIES_STORE = build_or_load_vectorstore("news", NEWS_DATA_PATH, "summaries")
_LEGAL_JUDGEMENTS_STORE = build_or_load_vectorstore("legal", LEGAL_DATA_PATH, "judgements")
_LEGAL_SUMMARIES_STORE = build_or_load_vectorstore("legal", LEGAL_DATA_PATH, "summaries")
print("[INIT] Vector stores ready.")

# --- Summarization Logic ---
SUMMARY_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a summarization assistant. Use the context to answer the question. Cite sources as [S1], [S2], etc."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
def _summarize(domain: str, query: str, subdir: str) -> str:
    store_map = {("news", "articles"): _NEWS_ARTICLES_STORE, ("news", "summaries"): _NEWS_SUMMARIES_STORE, ("legal", "judgements"): _LEGAL_JUDGEMENTS_STORE, ("legal", "summaries"): _LEGAL_SUMMARIES_STORE}
    store = store_map.get((domain, subdir))
    if not store: return f"Invalid store for domain '{domain}' or subdir '{subdir}'."
    retriever = store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)
    if not docs: return f"No relevant documents found for '{query}'."
    context_lines = [f"[S{i+1}] {d.page_content.replace('', ' ')}" for i, d in enumerate(docs)]
    context = "\n\n".join(context_lines)
    chain = SUMMARY_TEMPLATE | get_llm()
    response = chain.invoke({"question": query, "context": context})
    return response.content if hasattr(response, "content") else str(response)

# --- Tools ---
@tool("news_articles_summarizer")
def news_articles_summarizer_tool(query: str) -> str:
    """Summarizes news articles for a given query."""
    return _summarize("news", query, "articles")
@tool("news_summaries_summarizer")
def news_summaries_summarizer_tool(query: str) -> str:
    """Summarizes existing news summaries for a given query."""
    return _summarize("news", query, "summaries")
@tool("legal_judgements_summarizer")
def legal_judgements_summarizer_tool(query: str) -> str:
    """Summarizes legal judgements for a given query."""
    return _summarize("legal", query, "judgements")
@tool("legal_summaries_summarizer")
def legal_summaries_summarizer_tool(query: str) -> str:
    """Summarizes existing legal summaries for a given query."""
    return _summarize("legal", query, "summaries")
TOOLS = [news_articles_summarizer_tool, news_summaries_summarizer_tool, legal_judgements_summarizer_tool, legal_summaries_summarizer_tool]

# --- Agent Setup ---
original_prompt = hub.pull("hwchase17/react-chat")

custom_instructions = (
    "You are a specialized orchestrator agent. Your primary function is to analyze user queries and route them to either a 'news' or 'legal' domain tool. "
    "You must be concise and factual. When you provide the Final Answer, you must state which domain tool you used, for example: 'Based on the news domain...'. "
    "Do not provide opinions or information not present in the tool's output."
)

# print("[INFO] Modifying agent prompt with custom instructions...")
# print(f"[DEBUG] Original prompt messages: {original_prompt}")
original_system_template_string = original_prompt.template

# 4. Create a new, standalone SystemMessagePromptTemplate with your combined text.
system_message_template = SystemMessagePromptTemplate.from_template(
    custom_instructions + "\n\n" + original_system_template_string
)

# 5. Get all the *other* messages (placeholders, etc.) from the original prompt.
# other_messages = original_prompt.messages[1:]

# 6. Assemble a new ChatPromptTemplate from your new system message and the other messages.
agent_prompt = ChatPromptTemplate.from_messages([system_message_template])

_BASE_LLM = get_llm()
_REACT_AGENT = create_react_agent(_BASE_LLM, TOOLS, agent_prompt)
_EXECUTOR = AgentExecutor(agent=_REACT_AGENT, tools=TOOLS, verbose=True, handle_parsing_errors=True)

def run_agent(query: str) -> str:
    """Execute the agent with the given user query and return the final answer string."""
    result = _EXECUTOR.invoke({"input": query, "chat_history": []})
    return result.get("output", "<no output>")

# --- CLI Entrypoint ---
def _interactive_loop():
    print("\nAgentic RAG (news + legal) ready. Type a query or 'exit'.\n")
    while True:
        try:
            q = input("Query> ").strip()
            if not q: continue
            if q.lower() in {"exit", "quit"}: break
            answer = run_agent(q)
            print(f"\n--- Answer ---\n{answer}\n--------------\n")
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"[ERROR] {e}\n")
    print("Bye.")

if __name__ == "__main__":
    _interactive_loop()