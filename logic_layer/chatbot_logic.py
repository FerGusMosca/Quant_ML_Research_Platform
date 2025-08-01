import os
import re
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

FREEZE_SHARES_URL = "http://localhost:30903/Position/FreezeShares"
TRANSFER_SHARES_URL = "http://localhost:30903/Position/TransferShares"
DEFAULT_SOURCE_SHAREHOLDER_ID = 4

client = OpenAI()
embedding = OpenAIEmbeddings()
vectordb = FAISS.load_local("vectorstore", embeddings=embedding, index_name="index",
                            allow_dangerous_deserialization=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), memory=memory)

# Estado persistente por usuario (simplificado en memoria global)
current_intent_state = None


class IntentState:
    def __init__(self, amount=None, symbol=None, dest_id=None):
        self.amount = amount
        self.symbol = symbol
        self.dest_id = dest_id

    def is_complete(self):
        return self.amount is not None and self.symbol is not None and self.dest_id is not None

    def missing_slots(self):
        missing = []
        if self.amount is None: missing.append("amount")
        if self.symbol is None: missing.append("symbol")
        if self.dest_id is None: missing.append("destination_id")
        return missing


def detect_intent_and_slots_with_llm(question: str):
    prompt = f"""
You are an intent detector and slot extractor.

The user might be asking to transfer shares.
Extract the intent ("transfer_shares" or "other") and fill as many fields as possible.

Return JSON like:
{{
  "intent": "transfer_shares",
  "amount": 100,
  "symbol": "AAPL",
  "destination_id": 5
}}

If a field is missing, use null.

User input:
\"\"\"{question}\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception as e:
        return {"intent": "other"}


def freeze_then_transfer(amount: int, symbol: str, dest_shareholder_id: int) -> str:
    try:
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


def resolve_intent_flow(question: str):
    global current_intent_state

    parsed = detect_intent_and_slots_with_llm(question)

    if parsed.get("intent") != "transfer_shares":
        current_intent_state = None
        return None  # not an actionable intent

    if current_intent_state is None:
        current_intent_state = IntentState()

    # Fill detected slots
    if parsed.get("amount") is not None:
        current_intent_state.amount = parsed["amount"]
    if parsed.get("symbol") is not None:
        current_intent_state.symbol = parsed["symbol"].upper()
    if parsed.get("destination_id") is not None:
        current_intent_state.dest_id = parsed["destination_id"]

    if current_intent_state.is_complete():
        result = freeze_then_transfer(
            current_intent_state.amount,
            current_intent_state.symbol,
            current_intent_state.dest_id
        )
        current_intent_state = None  # reset
        return result
    else:
        missing = current_intent_state.missing_slots()
        pretty = ", ".join(missing)
        return f"🧠 I need more info to complete the transfer. Can you provide: {pretty}?"


def ask_bot(question: str) -> str:
    try:
        response = resolve_intent_flow(question)
        if response:
            return response
        else:
            rag_response = qa_chain.invoke({"question": question})
            return rag_response["answer"]
    except Exception as e:
        return f"💥 LLM Error: {str(e)}"
