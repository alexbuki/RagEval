# RAG Eval Setup

This project downloads the ConfRAG dataset, loads it into your lightrag server, and then lets you query the indexed dataset systematically using either vector based or relational graph based RAG for evaluation

## 1) Set up LightRAG

Clone the LightRAG project and set up a virtual environment:

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Run the minimal setup to choose endpoints for LLM and embedding model
```bash
make env-base 
```

Then start the LightRAG server with the default port:

```bash
lightrag-server
```

Make sure the server is running at `http://localhost:9621/webui` before continuing.

## 2) Download the dataset

```bash
python download_dataset.py
```

This downloads the dataset to:

```text
./data/ConfRAG.jsonl
```

## 3) Load the data

Once the dataset is downloaded, run:

```bash
python server_load.py
```

This script:

- optionally clears the current LightRAG dataset
- reads from `./data/ConfRAG.jsonl`
- sends the website *content* to LightRAG `/documents/text` endpoint

## 4) Query the data `TODO`


```bash
python probe.py
```

You can also use the LightRAG web UI to view the indexed data and query

## 4) Run Evaluation `TODO`