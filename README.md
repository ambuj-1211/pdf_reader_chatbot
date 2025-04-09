Prerequisite:
* Run ollama locally
* Set .env file with the following content
- OLLAMA_BASE_URL=http://host.docker.internal:11434 (if you are using ollama or else set the env file with api credentials for specific llm)
- LLM=llama2(or any llm you are using in ollama)
- EMBEDDING_MODEL=sentence_transformer

How to run:
- create virtual environment using `virtualenv venv`
- activate the environment using `source venv/bin/activate`
- install using requirements file `pip install -r requirements.txt`
- run the code `streamlit run pdf_bot.py`
