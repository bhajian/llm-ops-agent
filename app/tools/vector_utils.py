# --- app/tools/vector_utils.py ---
from typing import Iterable, Union
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
import weaviate

from app.config import get_settings

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)


def _get_db():
    cfg = get_settings()
    client = weaviate.Client(cfg["weaviate_url"])
    embeddings = (
        OpenAIEmbeddings()                              # if OPENAI_API_KEY present
        if "OPENAI_API_KEY" in os.environ
        else HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    return Weaviate(client, "Document", "content", embeddings)


def ingest_texts(texts: Iterable[str]) -> int:
    """
    Add a list of raw strings to the Weaviate index.
    Returns the number of chunks actually written.
    """
    docs = []
    for txt in texts:
        for chunk in _TEXT_SPLITTER.split_text(txt):
            docs.append({"content": chunk})

    if not docs:
        return 0

    db = _get_db()
    db.add_documents(docs)
    return len(docs)


def ingest_files(paths: Iterable[str]) -> int:
    """
    Convenience wrapper that loads (txt/pdf/…) files and pipes them to `ingest_texts`.
    """
    texts = []
    for path in paths:
        loader = TextLoader(path)
        docs = loader.load()
        texts.extend(d.page_content for d in docs)
    return ingest_texts(texts)
