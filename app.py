import os, json
from datetime import datetime
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from requests_oauthlib import OAuth2Session
from openai import OpenAI
from pypdf import PdfReader
import docx2txt

# ================= ENV =================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

USER_FILE = "users.json"
MEMORY_FILE = "memory.json"

for f in [USER_FILE, MEMORY_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            json.dump({}, fp)

# ================= HELPERS =================
def load_json(f):
    return json.load(open(f))

def save_json(f, d):
    json.dump(d, open(f, "w"), indent=2)

def ethics_guard(t):
    banned = ["kill","bomb","hack","rape","suicide","weapon"]
    return (False,"⚠️ Unsafe request.") if any(w in t.lower() for w in banned) else (True,"")

def update_memory(u,t):
    mem = load_json(MEMORY_FILE)
    mem.setdefault(u,{})
    if "my name is" in t.lower():
        mem[u]["name"] = t.split("is")[-1].strip()
    save_json(MEMORY_FILE, mem)
    return mem[u]

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

# ================= FASTAPI AUTH =================
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="dexora-session")

@app.get("/")
async def root(req: Request):
    return RedirectResponse("/chat") if "user" in req.session else RedirectResponse("/login")

@app.get("/login")
async def login():
    oauth = OAuth2Session(GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI,
        scope=["openid","email","profile"])
    auth,_ = oauth.authorization_url("https://accounts.google.com/o/oauth2/auth")
    return RedirectResponse(auth)

@app.get("/auth/google/callback")
async def callback(req: Request):
    oauth = OAuth2Session(GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI)
    oauth.fetch_token("https://oauth2.googleapis.com/token",
        client_secret=GOOGLE_CLIENT_SECRET,
        authorization_response=str(req.url))
    req.session["user"] = oauth.get(
        "https://www.googleapis.com/oauth2/v1/userinfo").json()
    return RedirectResponse("/chat")

# ================= STREAMING CHAT =================
def chat(msg, history, user, file):
    if history is None: history=[]
    ok, warn = ethics_guard(msg)
    if not ok:
        history.append((msg, warn))
        return history, ""

    mem = update_memory(user, msg)
    if file:
        msg += extract_text(file)

    messages = [{"role":"system","content":"You are Dexora, a smart AI."}]
    for u,a in history:
        messages += [{"role":"user","content":u},{"role":"assistant","content":a}]
    messages.append({"role":"user","content":msg})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )

    reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            reply += chunk.choices[0].delta.content
            yield history + [(msg, reply)], ""

# ================= UI + MOBILE CSS =================
css = """
body{background:#0f0f0f}
#chatbot{height:calc(100vh - 90px)}
.input-row{position:fixed;bottom:0;width:100%;
background:#0f0f0f;padding:8px;border-top:1px solid #222}
textarea{border-radius:18px!important;font-size:16px}
button{border-radius:50%!important;width:46px;height:46px}
"""

with gr.Blocks(css=css) as ui:
    gr.Markdown("## 🤖 Dexora")
    chatbox = gr.Chatbot(elem_id="chatbot")
    user = gr.State("User")

    with gr.Row(elem_classes="input-row"):
        msg = gr.Textbox(placeholder="Message Dexora...", scale=6)
        file = gr.File(label="📎", scale=1)
        send = gr.Button("➤", scale=1)

    send.click(chat, [msg, chatbox, user, file], [chatbox, msg])
    msg.submit(chat, [msg, chatbox, user, file], [chatbox, msg])

app = gr.mount_gradio_app(app, ui, path="/chat")
