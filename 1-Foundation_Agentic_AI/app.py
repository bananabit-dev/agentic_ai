from dotenv import load_dotenv
from pypdf import PdfReader
import requests
import json
from IPython.display import Markdown, display
import gradio as gr
from pathlib import Path
import os

#Settings
openrouter_base_url="https://openrouter.ai/api/v1"
model="z-ai/glm-4.5-air"

# Send the Mobile Notifications
PUSH_NOTIFICATION_URI="https://api.pushover.net/1/messages.json"
load_dotenv(override=True)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")

def push_notification(title,message):
    data={
        "token": pushover_token,
        "user": pushover_user,
        "message": message,
        "title": title,
    }

    response = requests.post(PUSH_NOTIFICATION_URI, data)
    if response.status_code == 200:
        return "Notification sent!"
    else:
        return f"Failed to send: {response.text}"

push_notification("Test Mobile Notification", "Testing via API calls")


# Define the tool functions


def record_user_details(email,name="Name not provided",notes="notes not provided"):
    push_notification("user record",f"[User Interest] {name} {email} | Notes: {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push_notification("unknown question",f"[Unknown Question] {question}")
    return {"recorded": "ok"}

# ---- Tool Schemas ----
record_user_details_json = {
    "name": "record_user_details",
    "description": "Record a user's interest using their email and optional details.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "User's email address"},
            "name": {"type": "string", "description": "User's name"},
            "notes": {"type": "string", "description": "Additional context or comments"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Log a question that the assistant couldn't answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unanswerable question"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

# ---- Tool Dispatcher ----

TOOL_FUNCTIONS = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question
}

def dispatch_tool_calls(tool_calls):
    """Execute tool calls and return their results."""
    tool_messages = []

    for call in tool_calls:
        # Handle both dict and OpenAI-style object
        if isinstance(call, dict):
            name = call.get("tool")
            args = call.get("arguments", {})
            call_id = call.get("id", "manual-call")
        else:
            name = call.function.name
            args = json.loads(call.function.arguments)
            call_id = getattr(call, "id", "manual-call")

        func = TOOL_FUNCTIONS.get(name)

        if func:
            try:
                tool_result = func(**args)
            except Exception as e:
                tool_result = {"error": str(e)}
        else:
            tool_result = {"error": f"unknown tool {name}"}

        tool_messages.append({
            "role": "tool",
            "content": json.dumps(tool_result),
            "tool_call_id": call_id
        })

    return tool_messages

# Test dispatch_tool_calls function
from types import SimpleNamespace

# --- Mock Tool Call Class (similar to OpenAI's tool_call format) ---
def make_tool_call(name, arguments, call_id="test-id"):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        id=call_id
    )

# --- Positive Test Case ---
tool_calls_positive =[
    make_tool_call("record_user_details", {"email": "anshulc55@icloud.com", "name": "Anshul Chauhan"}),
    make_tool_call("record_unknown_question", {"question": "What is the meaning of life?"}),
]

# --- Negative Test Case ---
tool_calls_negative = [
    make_tool_call("non_existent_tool", {"Dummy": "Test Dummy Value"}),
    make_tool_call("record_user_details", {})  # missing required "email"
]

# --- Run Tests ---
print("\n--- Positive Test Case ---")
print(dispatch_tool_calls(tool_calls_positive))

print("\n--- Negative Test Case ---")
print(dispatch_tool_calls(tool_calls_negative))

# Read the Profile PDF
pdfReader = PdfReader("..\\resources\\Profile.pdf")
prof_summary = ""
for page in pdfReader.pages:
    text = page.extract_text()
    if text:
        prof_summary += text

        import os

# Get the directory of the current script
script_dir = Path.cwd().parent

# Build the relative path from the script's directory
summ_filePath = os.path.join(script_dir, "resources", "Summary.txt")
with open(summ_filePath, "r", encoding="utf-8") as f:
    summary = f.read()

print(f"File path used: {summ_filePath}")  # Very helpful for debugging!

# System Prompt
name = "Sora"
system_prompt = (
    f"You are acting as {name}, representing {name} on their website. "
    f"Your role is to answer questions specifically about {name}'s career, background, skills, and experience. "
    f"You must faithfully and accurately portray {name} in all interactions. "
    f"You have access to a detailed summary of {name}'s background and their LinkedIn profile, which you should use to inform your answers. "
    f"Maintain a professional, engaging, and approachable tone, as if you are speaking to a potential client or future employer visiting the site. "
    f"Always record any unanswered questions using the record_unknown_question tool — even if the question seems minor or unrelated."
    f"If the user shows interest or continues chatting, encourage them to share their email address. Then, use the record_user_details tool to capture their email, name (if provided), and any context worth preserving."

    f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{prof_summary}\n\n"
    f"Using this context, please converse naturally and consistently, always staying in character as {name}."
)

import json
from openai import OpenAI

load_dotenv(override=True)

api_key=os.getenv("API_KEY")

# ✅ Create OpenAI-compatible client for OpenRouter
client = OpenAI(api_key=api_key, base_url=openrouter_base_url)

def chat_with_tools_openrouter(message, model,history, tools):
    """
    Handles a chat turn with OpenRouter, executes tools if called, and returns the final assistant message.
    """
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]

    # ✅ Ensure messages is a list of dicts
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError(f"Invalid messages format: {messages}")

    # ✅ Call the model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # ✅ If the model called a tool
    if message.tool_calls:
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            func = TOOL_FUNCTIONS.get(name)
            if func:
                result = func(**args)
                # Append tool result to messages
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
                # Call the model again with the tool result
                followup = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return followup.choices[0].message.content
    else:
        return message.content
    






def gradio_chat(message, history):    
    tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]
    return chat_with_tools_openrouter(
        message=message,
        model="z-ai/glm-4.5-air",  # must be a valid OpenRouter model
        history=history,
        tools=tools
    )

gr.ChatInterface(gradio_chat, type="messages").launch(share=True)