# 📄 PDF Chatbot with Streamlit & Ollama (LLM)

A simple Streamlit-based chatbot that allows you to interact with PDF content using a locally hosted LLM through Ollama.

---

## ✅ Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** and **Docker Compose**

---

### 📁 Step 1: Spin up the containers using docker compose

`docker compose up -d`

### Step 2: Verify Container Status:
`docker ps`

### Step 3: Pull LLM Model inside Ollama Container
1. `docker exec -it ollama bash`
2. Inside the container run: `ollama pull gemma2:2b`
3. Verify llm: `ollama list` it must be showing `gemma2:2b`
4. Exit container using `Ctrl + D`

## 🚀 Getting Started with Docker Compose

This project uses Docker Compose to manage and run both:
- A Streamlit app (on`http://localhost:8080/`)
- An Ollama container hosting the LLM (on `http://localhost:11434/`)
