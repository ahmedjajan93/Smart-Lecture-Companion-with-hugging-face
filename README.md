🎓 Smart Lecture Companion with Ollama:

A local-first AI-powered assistant that helps users understand, summarize, and interact with lecture content using LangChain and Ollama.

Built for students, educators, and researchers who want intelligent insights from their lecture material without depending on the cloud.

✨ Features:

🧠 Contextual Q&A — Ask deep, meaningful questions about your lecture content.

📝 Automatic Summarization — Extract concise summaries and bullet points.

🔗 LangChain Framework — Modular chains for parsing, memory, and reasoning.

🤖 Local LLM with Ollama — No API keys, no server — just fast, private processing.



📦 Tech Stack:

LangChain — for chain-based LLM workflows.

Ollama — to run LLMs locally (e.g. LLaMA, Mistral, or Gemma).

Streamlit — for a fast and interactive UI.

Python — core logic and orchestration.

⚙️ Installation:

Clone the repo: git clone https://github.com/ahmedjajan93/Smart-Lecture-Companion-with-ollama.git
cd smart-lecture-companion

Install dependencies:

pip install -r requirements.txt

Install & start Ollama:

ollama pull deepseek-r1:1.5b 

For embeddings:

ollama pull nomic-embed-text 


Start the Ollama server:
ollama run deepseek-r1:1.5b

Run the app:
streamlit run app.py

🧪 Usage:
1 - Upload your lecture file pdf.

2 - The app automatically extracts content and generates a summary.

3 - Ask follow-up questions in natural language.

Get intelligent, context-aware answers in real time.

🎥 Showcase:
✅ Example Workflow
🔹 Input: Lecture on Machine Learning
 - Upload: ml_lecture.pdf

🔹 Output:
. Summary:
"Covers supervised learning, overfitting, and model regularization."

. Questions:
Q: Why do decision trees overfit?
A: Because they can create overly complex splits that memorize training data.

🛠️ Customization:
🔧 Swap models in ollama run (e.g., llama2, gemma, mistral)

🧩 Extend with tools like vector search (e.g., ChromaDB or FAISS)

💬 Add memory for long conversations via LangChain's ConversationBufferMemory

📃 License:
MIT License — free to use, modify, and distribute.
