import os
import re
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

# Cargar variables de entorno desde .env
load_dotenv()

# Config
FREEZE_SHARES_URL = "http://localhost:30903/Position/FreezeShares"
TRANSFER_SHARES_URL = "http://localhost:30903/Position/TransferShares"
DEFAULT_SOURCE_SHAREHOLDER_ID = 4

# OpenAI client (nuevo SDK, sin pasar key explícitamente)
client = OpenAI()

# Vector store y modelo
embedding = OpenAIEmbeddings()
vectordb = FAISS.load_local("vectorstore", embeddings=embedding, index_name="index", allow_dangerous_deserialization=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), memory=memory)


def parse_transfer_intent(question: str):
    """
    Detecta si el mensaje expresa intención de transferir acciones.
    Extrae cantidad, símbolo y shareholder destino si aplica.
    """
    pattern = r"(?i)(?:send|transfer|pasale|mandale)\s+(\d+)\s+(?:shares\s+)?(\w+)\s+(?:to\s+)?(\d+)"
    match = re.search(pattern, question)
    if match:
        amount = int(match.group(1))
        symbol = match.group(2).upper()
        dest_id = int(match.group(3))
        return amount, symbol, dest_id
    return None


def freeze_then_transfer(amount: int, symbol: str, dest_shareholder_id: int) -> str:
    try:
        # Paso 1: Freezing
        freeze_payload = {
            "symbol": symbol,
            "physicalParticipantId": {
                "taxId": None,
                "shareholderId": DEFAULT_SOURCE_SHAREHOLDER_ID,
                "ssn": None
            },
            "amount": amount
        }
        freeze_resp = requests.post(FREEZE_SHARES_URL, json=freeze_payload)
        if not freeze_resp.ok:
            return f"❌ Error in freeze step: {freeze_resp.text}"

        # Paso 2: Transfer
        transfer_payload = {
            "symbol": symbol,
            "physicalSource": {
                "ssn": None,
                "shareholderId": DEFAULT_SOURCE_SHAREHOLDER_ID,
                "taxId": None
            },
            "physicalDestination": {
                "ssn": None,
                "shareholderId": dest_shareholder_id,
                "taxId": None
            },
            "amount": amount
        }
        transfer_resp = requests.post(TRANSFER_SHARES_URL, json=transfer_payload)
        if not transfer_resp.ok:
            return f"❌ Error in transfer step: {transfer_resp.text}"

        result = transfer_resp.json()
        return (
            f"✅ Shares transferred successfully!\n"
            f"🪪 Txn ID: {result.get('txnId')}\n"
            f"📄 Record ID: {result.get('recordId')}"
        )
    except Exception as e:
        return f"🔥 Internal error: {str(e)}"


def ask_bot(question: str) -> str:
    try:
        parsed = parse_transfer_intent(question)
        if parsed:
            amount, symbol, dest_id = parsed
            return freeze_then_transfer(amount, symbol, dest_id)

        response = qa_chain.invoke({"question": question})
        return response["answer"]
    except Exception as e:
        return f"💥 LLM Error: {str(e)}"
