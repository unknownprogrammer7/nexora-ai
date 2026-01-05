# -----------------------------
# IMPORTS
# -----------------------------
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
import speech_recognition as sr

# -----------------------------
# CONFIG
# -----------------------------
import os

GOOGLE_CLIENT_ID = os.getenv("701968721814-eof81fgbm6vl3keunbofdjthacu2h99a.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.getenv("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")
OPENAI_API_KEY = os.getenv("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")
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://dexora-ai.onrender.com/auth/google/callback")

client = OpenAI(api_key=OPENAI_API_KEY)
USER_FILE = "users.json"
MEMORY_FILE = "memory.json"

# -----------------------------
# INIT FILES
# -----------------------------
for f in [USER_FILE, MEMORY_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            json.dump({}, fp)

# -----------------------------
# HELPERS
# -----------------------------
def load_json(file):
    with open(file, "r") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

def ethics_guard(user_input):
    banned = ["kill","bomb","terrorist","hack bank","rape","suicide","drugs","weapon","murder"]
    for w in banned:
        if w in user_input.lower():
            return False, "⚠️ I cannot assist with harmful or illegal requests."
    return True, ""

def update_memory(username, user_input):
    memory = load_json(MEMORY_FILE)
    memory.setdefault(username, {})

    text = user_input.lower()
    if "my name is" in text:
        memory[username]["name"] = user_input.split("is")[-1].strip()
    if "i am from" in text:
        memory[username]["location"] = user_input.split("from")[-1].strip()

    save_json(MEMORY_FILE, memory)
    return memory[username]

def extract_text(file):
    if file is None: return ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file.name)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    if file.name.endswith(".docx"):
        return docx2txt.process(file.name)
    if file.name.endswith(".txt"):
        with open(file.name, "r", encoding="utf-8") as f:
            return f.read()
    return "Unsupported file format"

def voice_to_text(audio_file):
    if audio_file is None: return ""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return ""

# -----------------------------
# FASTAPI APP (Google OAuth)
# -----------------------------
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="dexora-secret-key")

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USERINFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"

@app.get("/")
async def root(request: Request):
    if "user" in request.session:
        return RedirectResponse("/chat")
    return RedirectResponse("/login")

@app.get("/login")
async def login_page():
    oauth = OAuth2Session(GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI, scope=["openid","email","profile"])
    auth_url, state = oauth.authorization_url(AUTH_URI, access_type="offline", prompt="select_account")
    return RedirectResponse(auth_url)

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    oauth = OAuth2Session(GOOGLE_CLIENT_ID, redirect_uri=REDIRECT_URI)
    token = oauth.fetch_token(TOKEN_URI, client_secret=GOOGLE_CLIENT_SECRET, authorization_response=str(request.url))
    resp = oauth.get(USERINFO_URI)
    user_info = resp.json()
    request.session["user"] = user_info
    return RedirectResponse("/chat")

# -----------------------------
# CHAT FUNCTION
# -----------------------------
def chat(user_message, history, username, uploaded_file=None):
    allowed, warning = ethics_guard(user_message)
    if not allowed:
        history.append((user_message, warning))
        return history, ""

    if history is None: history = []
    if not username: username = "User"

    user_mem = update_memory(username, user_message)

    # Handle uploaded file text
    if uploaded_file:
        file_text = extract_text(uploaded_file)
        user_message += f"\n[FILE CONTENT]: {file_text}"

    # Memory replies
    if "what is my name" in user_message.lower() and "name" in user_mem:
        reply = f"Your name is {user_mem['name']} 😊"
    elif "where am i from" in user_message.lower() and "location" in user_mem:
        reply = f"You are from {user_mem['location']} 🌍"
    elif "year" in user_message.lower():
        now = datetime.now()
        reply = f"The current year is {now.year}."
    else:
        messages = [{"role":"system","content":"You are Dexora, a clever AI assistant."}]
        for u,a in history:
            messages.append({"role":"user","content":u})
            messages.append({"role":"assistant","content":a})
        messages.append({"role":"user","content":user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content

    if "name" in user_mem:
        reply = f"{user_mem['name']}, {reply}"

    history.append((user_message, reply))
    return history, ""  # text-only reply

# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks() as chat_app:
    gr.Markdown("## Dexora AI")

    username_state = gr.State("User")
    chatbot = gr.Chatbot(elem_id="chatbox", height=400)

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message...", show_label=False, scale=2)
        send = gr.Button("Send", scale=1)
        voice = gr.Audio(source="microphone", type="filepath", label="🎤 Voice Input", scale=1)
        upload = gr.File(label="📎 Upload File", type="file", scale=1)

    # Actions
    voice.change(voice_to_text, voice, msg)
    send.click(chat, [msg, chatbot, username_state, upload], [chatbot, msg])
    msg.submit(chat, [msg, chatbot, username_state, upload], [chatbot, msg])

# -----------------------------
# Mount Gradio on FastAPI
# -----------------------------
app = gr.mount_gradio_app(app, chat_app, path="/chat")
