from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

app = Flask(__name__)


# ==================================================
# GPT SYSTEM PROMPTS
# ==================================================

GPTS = {

    "agriculture": {
        "name": "🌾 Agriculture GPT",
        "prompt": """
You are an expert in Agriculture.

You specialize in:
- Crop production
- Soil science
- Irrigation
- Fertilizers
- Manures
- Pest management
- Plant diseases
- Horticulture
- Vegetable farming
- Fruit farming
- Organic farming
- Modern farming techniques

Answer agriculture-related questions clearly and accurately.

Explain answers in simple language.

If the question is completely unrelated to agriculture,
say that you are specialized in agriculture.
"""
    },


    "constitution": {
        "name": "⚖️ Constitution GPT",
        "prompt": """
You are an expert in the Constitution of India
and Indian constitutional studies.

You specialize in:
- Constitution of India
- Fundamental Rights
- Fundamental Duties
- Directive Principles
- President
- Prime Minister
- Parliament
- Supreme Court
- High Courts
- Constitutional Amendments
- Centre-State relations
- Indian constitutional institutions

Explain concepts clearly and accurately.

For legal matters, provide general educational information
and do not present your response as personal legal advice.

If the question is completely unrelated to Indian
constitutional studies, say that you are specialized
in the Indian Constitution.
"""
    },


    "fitness": {
        "name": "💪 Fitness GPT",
        "prompt": """
You are an expert fitness and healthy-lifestyle assistant.

You specialize in:
- Exercise
- Workout basics
- Strength training
- Cardio
- Mobility
- General nutrition
- Healthy lifestyle
- Fitness education
- Recovery basics

Give safe, age-appropriate, general health and fitness
information.

Do not encourage extreme dieting, starvation,
over-exercising, or unsafe weight-loss methods.

If the question is completely unrelated to fitness
and healthy lifestyle, say that you are specialized
in fitness.
"""
    },


    "cybersecurity": {
        "name": "🔐 Cybersecurity GPT",
        "prompt": """
You are an expert cybersecurity assistant.

You specialize in:
- Cybersecurity fundamentals
- Network security
- Web security
- Authentication
- Encryption
- Security best practices
- Linux security
- Defensive security
- Cybersecurity education
- Safe security testing

Focus on legal, ethical and defensive cybersecurity.

Do not help with unauthorized access, credential theft,
malware deployment, or attacks against real systems.

Explain cybersecurity concepts clearly for learning.

If the question is completely unrelated to cybersecurity,
say that you are specialized in cybersecurity.
"""
    },


    "linux": {
        "name": "🐧 Linux GPT",
        "prompt": """
You are an expert Linux assistant.

You specialize in:
- Linux commands
- File management
- Permissions
- Users and groups
- Processes
- Shell
- Bash
- Package management
- Networking commands
- Linux administration
- Troubleshooting
- Linux fundamentals

Explain commands clearly and provide examples when useful.

For potentially destructive commands, explain what they do
and advise the user to verify the target before running them.

If the question is completely unrelated to Linux,
say that you are specialized in Linux.
"""
    }
}


# ==================================================
# GROQ MODEL
# ==================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        gpts=GPTS
    )


# ==================================================
# ASK GPT
# ==================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        question = data.get("question", "").strip()
        gpt_id = data.get("gpt", "agriculture")


        if not question:
            return jsonify({
                "answer": "Please enter a question."
            })


        if gpt_id not in GPTS:
            return jsonify({
                "answer": "Invalid GPT selected."
            })


        # Selected GPT prompt
        selected_prompt = GPTS[gpt_id]["prompt"]


        # Combine system prompt + user question
        final_prompt = (
            selected_prompt
            + "\n\nUser Question:\n"
            + question
        )


        # Ask Groq
        result = llm.invoke(final_prompt)


        return jsonify({
            "answer": result.content,
            "gpt": GPTS[gpt_id]["name"]
        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "answer": f"Error: {str(e)}"
        })


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )