import os, json, pickle, numpy as np
from datetime import datetime
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from requests_oauthlib import OAuth2Session
from openai import OpenAI
from pypdf import PdfReader
import docx2txt
import faiss

# ================= CONFIG =================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

client = OpenAI(api_key=OPENAI_API_KEY)

MEMORY_FILE = "memory.json"
VECTOR_FILE = "vector_store.pkl"
DIM = 1536

for f in [MEMORY_FILE]:
    if not os.path.exists(f):
        json.dump({}, open(f, "w"))

if os.path.exists(VECTOR_FILE):
    index, texts = pickle.load(open(VECTOR_FILE, "rb"))
else:
    index = faiss.IndexFlatL2(DIM)
    texts = []

# ================= HELPERS =================
def load_memory():
    return json.load(open(MEMORY_FILE))

def save_memory(d):
    json.dump(d, open(MEMORY_FILE, "w"), indent=2)

def ethics_guard(msg):
    banned = ["kill","bomb","hack","rape","suicide","weapon","murder"]
    return (False,"⚠️ Unsafe request.") if any(w in msg.lower() for w in banned) else (True,"")

def update_memory(user,msg):
    mem = load_memory()
    mem.setdefault(user,{})
    if "my name is" in msg.lower():
        mem[user]["name"] = msg.split("is")[-1].strip()
    save_memory(mem)
    return mem[user]

def extract_text(file):
    if not file: return ""
    if file.name.endswith(".pdf"):
        r = PdfReader(file.name)
        return "\n".join(p.extract_text() for p in r.pages if p.extract_text())
    if file.name.endswith(".docx"):
        return docx2txt.process(file.name)
    if file.name.endswith(".txt"):
        return file.read().decode()
    return ""

def embed(text):
    return np.array(client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding).astype("float32")

def save_vector_memory(text):
    vec = embed(text)
    index.add(vec.reshape(1,-1))
    texts.append(text)
    pickle.dump((index,texts), open(VECTOR_FILE,"wb"))

def recall_vector_memory(q):
    if index.ntotal == 0: return ""
    qv = embed(q)
    _, ids = index.search(qv.reshape(1,-1),2)
    return "\n".join(texts[i] for i in ids[0])

# ================= FASTAPI =================
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="dexora-secret-key")

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USERINFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"

@app.get("/")
async def root(req: Request):
    return RedirectResponse("/chat")

@app.get("/login")
async def login():
    oauth = OAuth2Session(
        GOOGLE_CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=["openid","email","profile"]
    )
    auth_url,_ = oauth.authorization_url(AUTH_URI, prompt="select_account")
    return RedirectResponse(auth_url)

@app.get("/auth/google/callback")
async def callback(req: Request):
    oauth = OAuth2Session(GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI)
    oauth.fetch_token(
        TOKEN_URI,
        client_secret=GOOGLE_CLIENT_SECRET,
        authorization_response=str(req.url)
    )
    req.session["user"] = oauth.get(USERINFO_URI).json()
    return RedirectResponse("/chat")

@app.get("/chat")
async def chat_guard(req: Request):
    if "user" not in req.session:
        return RedirectResponse("/login")
    return RedirectResponse("/chat/ui")

# ================= CHAT =================
def chat(msg, history, request: gr.Request, file):
    user = request.session["user"]["name"]
    if history is None: history=[]
    ok, warn = ethics_guard(msg)
    if not ok:
        history.append((msg,warn))
        return history,""

    mem = update_memory(user,msg)
    if file: msg += extract_text(file)

    recall = recall_vector_memory(msg)
    if recall: msg += f"\n[Memory]: {recall}"

    save_vector_memory(msg)

    messages = [{"role":"system","content":"You are Dexora AI."}]
    for u,a in history:
        messages += [{"role":"user","content":u},{"role":"assistant","content":a}]
    messages.append({"role":"user","content":msg})

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = r.choices[0].message.content
    history.append((msg, reply))
    return history,""

# ================= GRADIO =================
with gr.Blocks() as chat_app:
    gr.Markdown("## 💬 Dexora AI")
    chatbot = gr.Chatbot(height=450)
    with gr.Row():
        msg = gr.Textbox(placeholder="Message Dexora…", scale=8)
        upload = gr.File(scale=1)
        send = gr.Button("➤", scale=1)
    send.click(chat,[msg,chatbot,gr.Request(),upload],[chatbot,msg])
    msg.submit(chat,[msg,chatbot,gr.Request(),upload],[chatbot,msg])

app = gr.mount_gradio_app(app, chat_app, path="/chat/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT",8000)))
