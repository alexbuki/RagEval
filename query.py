import json
import requests

SERVER = "http://localhost:9621" # your running lightrag server
DATA_FILE = "./data/ConfRAG.jsonl" # make sure dataset is downloaded here successfuly
OUTPUT_FILE = "./data/ConfRAG_answers.jsonl"
MODES = ["local", "naive"]
NUM_RECORDS = 10 # the number of questions asked [SAME NUMBER AS FROM SERVER_LOAD.py]


with open(DATA_FILE, "r", encoding="utf-8") as file, open(OUTPUT_FILE, "w", encoding="utf-8") as output:
    
    for number in range(NUM_RECORDS):
        record = json.loads(file.readline())
        question = record["question"]

        for mode in MODES:
            response = requests.post(
                f"{SERVER}/query",
                json={"query": question, "mode": mode},
                timeout=512,
            )

            response.raise_for_status()

            result = {
                "id": record.get("id"),
                "question": question,
                "mode": mode,
                "response": response.json(),
            }

            output.write(json.dumps(result) + "\n")
            output.flush()

            print(f"{mode}: {number + 1}/{NUM_RECORDS}: {question}")

        