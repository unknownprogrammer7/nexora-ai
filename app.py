import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from pypdf import PdfReader
import docx2txt
import uvicorn
from openai import OpenAI

# =========================
# CONFIG / ENV
# =========================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_FILE = "chat_history.json"

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# APP INIT
# =========================
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# =========================
# GOOGLE OAUTH
# =========================
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# =========================
# HELPERS
# =========================
def load_chats():
    if not os.path.exists(CHAT_FILE):
        return {}
    with open(CHAT_FILE, "r") as f:
        return json.load(f)

def save_chats(data):
    with open(CHAT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def read_file(file: UploadFile):
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if file.filename.endswith(".docx"):
        return docx2txt.process(file.file)
    if file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    return "Unsupported file"

# Dummy role system
def get_role(email):
    # first user is admin, others user
    return "admin" if email.endswith("@admin.com") else "user"

# =========================
# ADMIN DASHBOARD
# =========================
@app.get("/admin")
async def admin_dashboard(request: Request):
    user = request.session.get("user")
    if not user or get_role(user["email"]) != "admin":
        return HTMLResponse("❌ Access Denied", status_code=403)

    chats = load_chats()
    return HTMLResponse(f"""
        <h2>Admin Dashboard</h2>
        <h3>Chat History</h3>
        <pre>{json.dumps(chats, indent=2)}</pre>
    """)

# =========================
# ROUTES
# =========================
@app.get("/")
async def home(request: Request):
    user = request.session.get("user")
    if not user:
        return HTMLResponse("""
            <h2>Dexora AI</h2>
            <a href="/login">Login with Google</a>
        """)

    chats = load_chats().get(user["email"], [])
    chat_html = "".join(
        f"<p><b>You:</b> {c['user']}<br><b>AI:</b> {c['assistant']}</p>"
        for c in chats
    )

    return HTMLResponse(f"""
    <html>
    <head>
      <link rel="manifest" href="/static/manifest.json">
    </head>
    <body>
      <h3>Welcome {user['email']}</h3>
      {chat_html}
      <form method="post" action="/chat">
        <input name="message" required>
        <button type="submit">Send</button>
      </form>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Upload File</button>
      </form>
      <a href="/logout">Logout</a>
    </body>
    </html>
    """)

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    user = request.session["user"]
    chats = load_chats()
    chats.setdefault(user["email"], [])

    # Call OpenAI GPT
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}]
    )
    reply = r.choices[0].message.content

    # Save chat
    chats[user["email"]].append({
        "user": message,
        "assistant": reply,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    save_chats(chats)
    return RedirectResponse("/", status_code=302)

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    user = request.session["user"]
    text = read_file(file)
    chats = load_chats()
    chats.setdefault(user["email"], [])
    chats[user["email"]].append({
        "user": f"Uploaded file: {file.filename}",
        "assistant": text[:500],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_chats(chats)
    return RedirectResponse("/", status_code=302)

@app.get("/login")
async def login(request: Request):
    return await oauth.google.authorize_redirect(request, request.url_for("auth"))

@app.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    request.session["user"] = token["userinfo"]
    return RedirectResponse("/")

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)

# =========================
# START SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
