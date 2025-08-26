# chatbot_app.py
import re
import os
import base64
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from openai import OpenAI   # ✅ Use OpenAI official client
import uuid

# -----------------------------
# Config / ENV
# -----------------------------
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB
client = MongoClient(mongo_uri)
DB_NAME = "FIRE"
COLLECTIONS = [
    "networths", "personalrisks", "net_worths", "multiusers", "mfdetails",
    "marriagefundplans", "insurances", "houseplans", "googles", "fundallocations",
    "childexpenses", "childeducations", "budgetincomeplans", "firequestions",
    "financials", "customplans", "expensesmasters", "vehicles", "emergencyfunds",
    "profiles", "realitybudgetincomes"
]
CHAT_COLLECTION = "chat_history"

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(page_title="FIFP", layout="centered")
st.markdown("""<style>.main { background-color: #f8f9fa; }</style>""", unsafe_allow_html=True)

# ✅ Style fixes for chat input + menu button
st.markdown("""
    <style>
    div.stChatInput {
        border: 2px solid #07771b;
        border-radius: 10px;
        padding: 5px;
        margin-top: 10px;
    }
    .floating-btn {
        position: fixed;
        top: 15px;
        right: 15px;
        background: #07771b;
        color: white;
        border: none;
        padding: 6px 14px;
        border-radius: 8px;
        cursor: pointer;
        z-index: 2000;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
    }
    .menu-btn {
        border: 1px solid #ff4b4b;
        background: transparent;
        color: #ff4b4b;
        border-radius: 6px;
        padding: 2px 6px;
        font-size: 18px;
        cursor: pointer;
    }
    .menu-popup {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 6px;
        margin-top: 5px;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.15);
    }
    .menu-popup button {
        display: block;
        width: 100%;
        text-align: left;
        padding: 6px 10px;
        margin: 4px 0;
        border: 1px solid #ddd;
        border-radius: 6px;
        background: #f8f9fa;
        cursor: pointer;
        font-size: 14px;
    }
    .menu-popup button:hover {
        background: #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

logo_base64 = get_base64_image("fifp_logo.png")
if logo_base64:
    st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{logo_base64}" width="60" height="60"/>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; font-size: 15px; color: #07771b;margin-top:10px'>Financial Independence Focus Passion</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;font-size:13px'>Chat with your personalized data</p>", unsafe_allow_html=True)

# === UserID Bridge ===
components.html("""
<script>
window.addEventListener('message', (event) => {
    const { userId } = event.data;
    if (userId) {
        localStorage.setItem('userID', userId);
        console.log('userID saved in iframe localStorage:', userId);
    }
});
document.addEventListener("DOMContentLoaded", function () {
    const userID = localStorage.getItem("userID");
    const streamlitReceiver = window.parent;
    if (streamlitReceiver) {
        streamlitReceiver.postMessage({ type: "userID", value: userID }, "*");
    }
});
</script>
""", height=0)

# Get userID safely (strip whitespace/newlines)
userID = st_javascript("localStorage.getItem('userID');")

query_params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
if query_params:
    user_input = query_params.get("userID", [userID])
else:
    user_input = userID

if user_input:
    user_input = str(user_input).strip()   # ✅ remove \n and spaces

# -----------------------------
# OpenAI client
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Helpers
# -----------------------------
def cosine_sim(a, b_matrix):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b_matrix / (np.linalg.norm(b_matrix, axis=1, keepdims=True) + 1e-10)
    return np.dot(b, a)

def embed_texts_openai(texts, model="text-embedding-3-small"):
    resp = openai_client.embeddings.create(model=model, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vectors

def generate_answer_openai(context, question, model="gpt-4o-mini"):
    system_prompt = (
        "You are a helpful personal finance assistant. "
        "ONLY use the provided CONTEXT (from the user's database). "
        "If the answer is not present, reply exactly with NO_ANSWER. "
        "Do not invent facts. Mask sensitive info (phones, cards, emails)."
    )
    user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nAnswer:"
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# Data loading
# -----------------------------
def load_user_documents(user_id_str):
    db = client[DB_NAME]
    all_docs = []
    for collection_name in COLLECTIONS:
        try:
            cursor = db[collection_name].find({"userId": ObjectId(user_id_str)})
            for doc in cursor:
                doc.pop("_id", None)
                doc.pop("userId", None)
                text = f"[{collection_name}]\n" + "\n".join([f"{k}: {v}" for k, v in doc.items()])
                all_docs.append(Document(page_content=text))
        except Exception as e:
            st.warning(f"⚠️ Error loading `{collection_name}`: {e}")
    return all_docs

def build_local_retriever(_docs):
    texts = [d.page_content for d in _docs]
    vectors = embed_texts_openai(texts)
    return {"docs": _docs, "embeddings": vectors}

# -----------------------------
# Masking helpers
# -----------------------------
def mask_sensitive(text: str) -> str:
    text = re.sub(r'([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', r'\1***\2', text)
    text = re.sub(r'(\+?\d[\d\-\s]{8,}\d)', lambda m: "****" + m.group(0)[-4:], text)
    text = re.sub(r'((?:\d[ -]?){13,19})', lambda m: "**** **** **** " + re.sub(r'\D', '', m.group(0))[-4:], text)
    return text

# -----------------------------
# Answer logic
# -----------------------------
def answer_from_db_local(retriever_obj, question, top_k=3, sim_threshold=0.12):
    doc_embeddings = retriever_obj["embeddings"]
    docs = retriever_obj["docs"]

    qvec = embed_texts_openai([question])[0]
    sims = cosine_sim(qvec, doc_embeddings)

    top_idx = np.argsort(-sims)[:top_k]
    top_scores = sims[top_idx]
    top_texts = [docs[i].page_content for i in top_idx]

    if len(top_scores) == 0 or float(np.max(top_scores)) < sim_threshold:
        return None, "❌ I couldn't find that in your data. Please ask about your budgets, investments, insurance, or other saved details."

    context = "\n\n".join(top_texts)[:8000]
    raw = generate_answer_openai(context=context, question=question)

    if "NO_ANSWER" in raw:
        return None, "❌ I couldn't find that in your data."

    return mask_sensitive(raw), None

# -----------------------------
# Default friendly answers
# -----------------------------
DEFAULT_RESPONSES = {
    "hi": "👋 Hi there! How can I help you with your finances today?",
    "hello": "👋 Hello! I'm your personal finance assistant. What would you like to know?",
    "hai": "😊 Hai! Ask me anything about your budgets, savings, or investments.",
    "hellow": "🙌 Hello! How can I guide you today?",
    "how are you": "🤗 I'm doing great, thank you! I'm here to help you with your financial journey.",
    "good morning": "🌅 Good morning! Ready to plan your financial goals today?",
    "good evening": "🌆 Good evening! Let's check your financial progress.",
    "thanks": "🙏 You're welcome! Happy to help anytime.",
    "bye": "👋 Goodbye! Take care of your finances."
}

def check_default_response(user_text: str):
    text = user_text.strip().lower()
    for key, resp in DEFAULT_RESPONSES.items():
        if key in text:
            return resp
    return None

# -----------------------------
# Chat History Persistence
# -----------------------------
def save_chat_to_mongo(user_id, messages, title):
    db = client[DB_NAME]
    chat_col = db[CHAT_COLLECTION]
    if messages:
        chat_doc = {
            "userId": ObjectId(user_id),
            "messages": messages,
            "title": title,
            "created_at": datetime.utcnow()
        }
        return chat_col.insert_one(chat_doc).inserted_id

def load_past_chats(user_id):
    db = client[DB_NAME]
    chat_col = db[CHAT_COLLECTION]
    return list(chat_col.find({"userId": ObjectId(user_id)}).sort("created_at", -1))

def delete_chat(chat_id):
    db = client[DB_NAME]
    chat_col = db[CHAT_COLLECTION]
    chat_col.delete_one({"_id": ObjectId(chat_id)})

def rename_chat(chat_id, new_title):
    db = client[DB_NAME]
    chat_col = db[CHAT_COLLECTION]
    chat_col.update_one({"_id": ObjectId(chat_id)}, {"$set": {"title": new_title}})

# -----------------------------
# UI
# -----------------------------
if user_input:
    status = st.empty()
    try:
        raw_docs = load_user_documents(user_input)

        if not raw_docs:
            status.error("❌ No documents found for this userId.")
        else:
            if "retriever_obj" not in st.session_state or st.session_state.get("retriever_user") != user_input:
                st.session_state.retriever_obj = build_local_retriever(raw_docs)
                st.session_state.retriever_user = user_input
            retriever_obj = st.session_state.retriever_obj

            # === Sidebar: Past Chats === 
            st.sidebar.header("📜 Previous Chats")
            past_chats = load_past_chats(user_input)
            
            # Initialize session state for menu and rename mode
            if "menu_open" not in st.session_state:
                st.session_state.menu_open = {}
            if "rename_mode" not in st.session_state:
                st.session_state.rename_mode = {}
                
            for chat in past_chats:
                chat_id = str(chat["_id"])
                col1, col2 = st.sidebar.columns([5, 1])
                
                with col1:
                    if st.button(chat["title"], key=f"title_{chat_id}"):
                        st.session_state.messages = chat["messages"]
                        st.session_state.current_chat_id = chat_id
                        
                with col2:
                    if st.button("⋮", key=f"menu_btn_{chat_id}"):
                        st.session_state.menu_open[chat_id] = not st.session_state.menu_open.get(chat_id, False)
                
                # Handle menu options
                if st.session_state.menu_open.get(chat_id, False):
                    with st.sidebar.container():
                        if st.session_state.rename_mode.get(chat_id, False):
                            new_title = st.text_input("✏️ Rename chat", value=chat["title"], key=f"rename_input_{chat_id}")
                            if st.button("✅ Save", key=f"save_rename_{chat_id}"):
                                rename_chat(chat_id, new_title)
                                st.session_state.rename_mode[chat_id] = False
                                st.rerun()
                        else:
                            if st.button("✏️ Rename", key=f"rename_{chat_id}"):
                                st.session_state.rename_mode[chat_id] = True
                                st.rerun()
                            if st.button("🗑️ Delete", key=f"delete_{chat_id}"):
                                delete_chat(chat_id)
                                st.session_state.menu_open.pop(chat_id, None)
                                st.rerun()
                            if st.button("🔗 Share", key=f"share_{chat_id}"):
                                st.sidebar.info(f"Share link: https://yourapp.com/chat/{chat_id}")

            # === Floating New Chat button ===
            if st.session_state.get("current_chat_id"):
                if st.button("➕ New Chat", key="new_chat", help="Start a new chat", type="primary"):
                    st.session_state.messages = []
                    st.session_state.current_chat_id = None
                    st.rerun()
                st.markdown('<button class="floating-btn" onclick="window.location.reload()">➕ New Chat</button>', unsafe_allow_html=True)

            # ✅ Init new chat session
            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.current_chat_id = None

            st.markdown("<div style='text-align: center; font-size: 15px; color: #07771b;'>💬 Ask me anything about your details:</div>", unsafe_allow_html=True)

            chat_container = st.container()

            # Show suggestions for new chats
            if not st.session_state.messages:
                suggestions = [
                    "What is my monthly budget?",
                    "Give me a summary of my investments",
                    "What is my emergency fund status?"
                ]
                cols = st.columns(len(suggestions))
                for i, s in enumerate(suggestions):
                    if cols[i].button(s, key=f"suggestion_{i}"):
                        st.session_state.messages.append({"role": "user", "content": s})
                        with st.spinner("🤔 Thinking..."):
                            default_reply = check_default_response(s)
                            if default_reply:
                                answer, err = default_reply, None
                            else:
                                answer, err = answer_from_db_local(retriever_obj, s)
                            if answer:
                                answer = (answer.replace("$", "₹")
                                                 .replace("USD", "INR")
                                                 .replace("usd", "INR")
                                                 .replace("dollars", "rupees")
                                                 .replace("Dollars", "Rupees"))
                        st.session_state.messages.append({"role": "assistant", "content": answer if answer else err})

            # Display chat messages
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Handle new user input
            prompt = st.chat_input("Type your question here...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container.chat_message("user"):
                    st.markdown(prompt)

                default_reply = check_default_response(prompt)
                if default_reply:
                    answer, err = default_reply, None
                else:
                    with st.spinner("🤔 Thinking..."):
                        answer, err = answer_from_db_local(retriever_obj, prompt)
                    if answer:
                        answer = (answer.replace("$", "₹")
                                         .replace("USD", "INR")
                                         .replace("usd", "INR")
                                         .replace("dollars", "rupees")
                                         .replace("Dollars", "Rupees"))
                
                st.session_state.messages.append({"role": "assistant", "content": answer if answer else err})
                with chat_container.chat_message("assistant"):
                    st.markdown(answer if answer else err)

            # Save chat to database
            if st.session_state.current_chat_id is None and st.session_state.messages:
                title = st.session_state.messages[0]["content"][:30]
                save_chat_to_mongo(user_input, st.session_state.messages, title)
                st.session_state.current_chat_id = "saved"

            elif st.session_state.current_chat_id not in (None, "saved"):
                db = client[DB_NAME]
                chat_col = db[CHAT_COLLECTION]
                chat_col.update_one(
                    {"_id": ObjectId(st.session_state.current_chat_id)},
                    {"$set": {"messages": st.session_state.messages}}
                )

    except Exception as e:
        status.error(f"❌ Error occurred: {str(e)}")
else:
    st.info("Waiting......")