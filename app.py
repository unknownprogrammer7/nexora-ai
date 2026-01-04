# -----------------------------
# IMPORTS
# -----------------------------
import gradio as gr
from pypdf import PdfReader
import docx2txt
import json
import os
from openai import OpenAI
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
client = OpenAI(api_key=os.getenv("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"))
USER_FILE = "users.json"
MEMORY_FILE = "memory.json"

# -----------------------------
# INIT FILES
# -----------------------------
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump({}, f)

# -----------------------------
# HELPERS
# -----------------------------
def load_json(file):
    with open(file, "r") as f:
        return json.load(f)

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# ETHICS GUARD
# -----------------------------
def ethics_guard(user_input):
    banned_keywords = [
        "kill", "bomb", "terrorist", "hack bank", "rape",
        "suicide", "drugs", "weapon", "murder"
    ]
    for word in banned_keywords:
        if word in user_input.lower():
            return False, (
                "⚠️ I cannot assist with harmful, illegal, or unethical requests."
            )
    return True, ""

# -----------------------------
# AUTH FUNCTIONS
# -----------------------------
def signup(username, password):
    users = load_json(USER_FILE)
    if username in users:
        return "❌ User already exists", gr.update(visible=True), gr.update(visible=False)
    users[username] = password
    save_json(USER_FILE, users)
    return "✅ Signup successful! Please login.", gr.update(visible=True), gr.update(visible=False)

def login(username, password):
    users = load_json(USER_FILE)
    if users.get(username) == password:
        return gr.update(visible=False), gr.update(visible=True), f"👋 Welcome {username}", username
    return gr.update(visible=True), gr.update(visible=False), "❌ Invalid login", ""

# -----------------------------
# MEMORY FUNCTIONS
# -----------------------------
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
    if file is None:
        return ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file.name)
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    if file.name.endswith(".docx"):
        return docx2txt.process(file.name)
    if file.name.endswith(".txt"):
        with open(file.name, "r", encoding="utf-8") as f:
            return f.read()
    return "Unsupported file format"

# -----------------------------
# CHAT FUNCTION
# -----------------------------
def chat(user_message, history, username):
    if history is None or not isinstance(history, list):
        history = []
    if not username:
        username = "User"

    # Ethics check
    allowed, warning = ethics_guard(user_message)
    if not allowed:
        history.append((user_message, warning))
        return history, ""

    # Update memory
    user_mem = update_memory(username, user_message)

    # Memory replies
    if "what is my name" in user_message.lower() and "name" in user_mem:
        reply = f"Your name is {user_mem['name']} 😊"
    elif "where am i from" in user_message.lower() and "location" in user_mem:
        reply = f"You are from {user_mem['location']} 🌍"
    elif "year" in user_message.lower():
        now = datetime.now()
        reply = f"The current year is {now.year}."
    else:
        # OpenAI response
        messages = [{"role": "system", "content": "You are Nexora, a smart AI assistant."}]
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content

    # Personal touch
    if "name" in user_mem:
        reply = f"{user_mem['name']}, {reply}"

    history.append((user_message, reply))
    return history, ""  # Only text

# -----------------------------
# CUSTOM CSS
# -----------------------------
custom_css = """
body {
    background: #0f172a;
}
.gradio-container {
    max-width: 100%;
}
#chatbox {
    background: #ffffff;
    border-radius: 12px;
}
textarea {
    background: #020617 !important;
    color: white !important;
    border-radius: 12px !important;
}
button {
    border-radius: 10px !important;
    font-weight: bold;
}
"""

# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    gr.Markdown("##  Nexora AI")

    username_state = gr.State("")

    # LOGIN PANEL
    with gr.Column(visible=True) as login_panel:
        gr.Markdown("###  Login ")
        lu = gr.Textbox(label="Username")
        lp = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_msg = gr.Text()

        gr.Markdown("### Sign Up")
        su = gr.Textbox(label="New Username")
        sp = gr.Textbox(label="New Password", type="password")
        signup_btn = gr.Button("Create Account")
        signup_msg = gr.Text()

    # CHAT PANEL
    with gr.Column(visible=False) as chat_panel:
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(elem_id="chatbox", height=400)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=2, show_label=False)
                send = gr.Button("Send", scale=1)

    # BUTTON ACTIONS
    signup_btn.click(signup, [su, sp], [signup_msg, login_panel, chat_panel])
    login_btn.click(login, [lu, lp], [login_panel, chat_panel, login_msg, username_state])
    send.click(chat, [msg, chatbot, username_state], [chatbot, msg])
    msg.submit(chat, [msg, chatbot, username_state], [chatbot, msg])

# -----------------------------
# LAUNCH
# -----------------------------
app.launch(server_name="0.0.0.0", server_port=10000)
