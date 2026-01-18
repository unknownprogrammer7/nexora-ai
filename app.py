import os
import json
import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from requests_oauthlib import OAuth2Session
from openai import OpenAI
from pypdf import PdfReader
import docx2txt
import faiss
import gradio as gr

# ================= CONFIG =================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SESSION_SECRET = os.getenv("SESSION_SECRET", "super-secret-key")

client = OpenAI(api_key=OPENAI_API_KEY)

MEMORY_FILE = "memory.json"
VECTOR_FILE = "vector_store.pkl"
CHAT_HISTORY_FILE = "chat_history.json"
DIM = 1536

# ================= INITIALIZATION =================
def load_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

for f in [MEMORY_FILE, CHAT_HISTORY_FILE]:
    if not os.path.exists(f):
        save_json(f, {})

if os.path.exists(VECTOR_FILE):
    index, texts = pickle.load(open(VECTOR_FILE, "rb"))
else:
    index = faiss.IndexFlatL2(DIM)
    texts = []

# ================= HELPERS =================
def ethics_guard(msg):
    banned = ["kill","bomb","hack","rape","suicide","weapon","murder"]
    return (False, "⚠️ Unsafe request.") if any(w in msg.lower() for w in banned) else (True, "")

def update_memory(user, msg):
    mem = load_json(MEMORY_FILE)
    mem.setdefault(user, {})
    if "my name is" in msg.lower():
        mem[user]["name"] = msg.split("is")[-1].strip()
    save_json(MEMORY_FILE, mem)
    return mem[user]

def extract_text(file):
    if not file:
        return ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file.name)
        return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    elif file.name.endswith(".docx"):
        return docx2txt.process(file.name)
    elif file.name.endswith(".txt"):
        return file.read().decode()
    return ""

def embed(text):
    return np.array(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    ).astype("float32")

def save_vector_memory(text):
    vec = embed(text)
    index.add(vec.reshape(1, -1))
    texts.append(text)
    pickle.dump((index, texts), open(VECTOR_FILE, "wb"))

def recall_vector_memory(q):
    if index.ntotal == 0:
        return ""
    qv = embed(q)
    _, ids = index.search(qv.reshape(1, -1), 2)
    return "\n".join(texts[i] for i in ids[0])

def load_chat_history():
    return load_json(CHAT_HISTORY_FILE)

def save_chat_history(data):
    save_json(CHAT_HISTORY_FILE, data)

# ================= FASTAPI =================
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USERINFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"

@app.get("/")
async def root(req: Request):
    return RedirectResponse("/chat/ui")

@app.get("/login")
async def login():
    oauth = OAuth2Session(
        GOOGLE_CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=["openid", "email", "profile"]
    )
    auth_url, _ = oauth.authorization_url(AUTH_URI, prompt="select_account")
    return RedirectResponse(auth_url)

@app.get("/auth/google/callback")
async def callback(req: Request):
    oauth = OAuth2Session(GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI)
    oauth.fetch_token(
        TOKEN_URI,
        client_secret=GOOGLE_CLIENT_SECRET,
        authorization_response=str(req.url)
    )
    userinfo = oauth.get(USERINFO_URI).json()
    req.session["user"] = userinfo
    return RedirectResponse("/chat/ui")

@app.get("/logout")
async def logout(req: Request):
    req.session.clear()
    return RedirectResponse("/login")

# ================= CHAT =================
def chat_function(msg, history, session_user, file):
    user = session_user["name"]
    history = history or []

    ok, warn = ethics_guard(msg)
    if not ok:
        history.append((msg, warn))
        return history, ""

    if file:
        msg += extract_text(file)

    update_memory(user, msg)

    recall = recall_vector_memory(msg)
    if recall:
        msg += f"\n[Memory]: {recall}"

    save_vector_memory(msg)

    messages = [{"role": "system", "content": "You are Dexora AI."}]
    for u, a in history:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": msg})

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = r.choices[0].message.content
    history.append((msg, reply))

    # Save history
    all_history = load_chat_history()
    user_history = all_history.get(user, [])
    user_history.append({"user": msg, "assistant": reply})
    all_history[user] = user_history
    save_chat_history(all_history)

    return history, ""

# ================= GRADIO UI =================
def get_session_user(req: Request):
    return req.session.get("user", None)

with gr.Blocks(css="static/style.css") as chat_app:
    gr.Markdown("## 💬 Dexora AI (Google Login Only)")

    login_btn = gr.Button("Login with Google")
    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="Message Dexora…", scale=8)
    upload = gr.File(scale=1)
    send = gr.Button("➤", scale=1)

    # Function to dynamically get session user from FastAPI
    def wrapper_chat(msg, history, file):
        from fastapi import Request
        req = gr.get_state("fastapi_request")
        user = get_session_user(req)
        if not user:
            return history or [], ""
        return chat_function(msg, history, user, file)

    send.click(wrapper_chat, [msg, chatbot, upload], [chatbot, msg])
    msg.submit(wrapper_chat, [msg, chatbot, upload], [chatbot, msg])

# Mount Gradio
app = gr.mount_gradio_app(app, chat_app, path="/chat/ui")

# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
