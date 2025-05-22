from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import weaviate
import os

def get_rag_chain():
    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    db = Weaviate(client, "Document", "content", OpenAIEmbeddings())
    retriever = db.as_retriever()

    llm = ChatOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://vllm.yourdomain.com/v1"),
        api_key="fake-api-key",  # Required but ignored by vLLM
        model_name="llama3-70b-chat",  # or the model you're running
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
