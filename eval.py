import json
import requests

SERVER = "http://localhost:9621"  # your running lightrag server
DATA_FILE = "./data/ConfRAG.jsonl"  # make sure dataset is downloaded here successfully
ANSWERS_FILE = "./data/ConfRAG_answers.jsonl"
MODES = ["local", "naive"]
NUM_RECORDS = 20  # the number of questions asked [SAME NUMBER AS FROM SERVER_LOAD.py]
OLLAMA_SERVER = "http://localhost:11434"
LLM_MODEL = "qwen2.5:14b"


def SemSim_LLMJudge():
    '''llm as a judge for semantic similarity'''
    prompt = (
        f"Your task is "
    )
    return False


def BinaryViewpoint_LLMJudge(question, viewpoint, response):
    '''llm as a judge to establish whether a response covers a viewpoint'''
    prompt = (
        f"Using the original query as reference\n"
        f"Your task is binary classification whether the *Generated Answer* covers the *Expected Viewpoint*\n"
        f"Original Query: {question}\n"
        f"Expected Viewpoint: {viewpoint}\n"
        f"Generated Answer: {response}\n"
        f"'YES' or 'NO', does the Generated Answer cover the Expected Viewpoint?"
        f"You must only respond 'YES' or 'NO', do not include anything else in the response"
    )
    try:
        r = requests.post(
            f"{OLLAMA_SERVER}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=60,
        ).json()
        print(r["response"])
        return "YES" in r["response"].upper()
    except Exception:
        print("ERROR")
        return False


ground_truth = {}
with open(DATA_FILE, "r", encoding="utf-8") as gt:
    for number in range(NUM_RECORDS):
        record = json.loads(gt.readline())
        ground_truth[record["id"]] = record
# print(ground_truth)
with open(ANSWERS_FILE, "r", encoding="utf-8") as file:
    correct = 0
    for number in range(NUM_RECORDS * len(MODES)):
        record = json.loads(file.readline())
        id = record["id"]
        gold = ground_truth.get(id)
        if not gold:
            continue

        question = record["question"]
        response = record["response"]["response"]
        mode = record["mode"]
        contradicts = gold["contradicts"]

        for item in (gold.get("answers", [])):
            viewpoint = item["answer"]
            is_covered = (BinaryViewpoint_LLMJudge(question, viewpoint, response))
            print(f"[{mode}] Q{id} Contradicts=[{contradicts}] viewpoint: {is_covered}")

