import json
import requests

SERVER = "http://localhost:9621"  # your running lightrag server
DATA_FILE = "./data/ConfRAG.jsonl"  # make sure dataset is downloaded here successfully
ANSWERS_FILE = "./data/ConfRAG_answers.jsonl"
MODES = ["hybrid", "naive"]
NUM_RECORDS = 50  # the number of questions asked [SAME NUMBER AS FROM SERVER_LOAD.py]
OLLAMA_SERVER = "http://localhost:11434"
LLM_MODEL = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS"


def SemSim_LLMJudge(question, viewpoint, response):
    '''llm as a judge for semantic similarity score 1 to 5'''
    prompt = (
        f"Using the original query as reference\n"
        f"Your task is to critically evaluate from 1 to 5 how well the *Generated Answer* covers the *Expected Viewpoint*\n"
        f"1 = completely missing, 3 = partially covered, 5 = fully covered\n"
        f"Original Query: {question}\n"
        f"Expected Viewpoint: {viewpoint}\n"
        f"Generated Answer: {response}\n"
        f"You must only respond with an integer from 1 to 5, do not include anything else in the response"
    )
    try:
        r = requests.post(
            f"{OLLAMA_SERVER}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}, "think": False},
            timeout=60,
        ).json()
        print(r["response"])
        for ch in r["response"]:
            if ch in "12345":
                return int(ch)
        return 1
    except Exception:
        print("ERROR")
        return 1


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
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}, "think": False},
            timeout=60,
        ).json()
        print(r["response"])
        return "YES" in r["response"].upper()
    except Exception:
        print("ERROR")
        return False


def ConflictBehavior_LLMJudge(question, response):
    '''llm as a judge to classify conflict behavior into SYNTHESIZED, ONE_SIDED, or BLENDED'''
    prompt = (
        f"Using the original query as reference\n"
        f"Your task is to classify how the *Generated Answer* handles conflicting evidence:\n"
        f"Z) SYNTHESIZED (explains multiple differing viewpoints)\n"
        f"Y) ONE_SIDED (favors only one viewpoint and ignores others)\n"
        f"X) BLENDED (blends contradictory facts into a false compromise)\n"
        f"Original Query: {question}\n"
        f"Generated Answer: {response}\n"
        f"You must only respond 'Z', 'Y', or 'X', do not include anything else in the response"
    )
    try:
        r = requests.post(
            f"{OLLAMA_SERVER}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}, "think": False},
            timeout=60,
        ).json()
        print(r["response"])
        resp = r["response"].upper()
        if "Z" in resp: return "SYNTHESIZED"
        if "Y" in resp: return "ONE_SIDED"
        if "X" in resp: return "BLENDED"
        return "UNKNOWN"
    except Exception:
        print("ERROR")
        return "ERROR"


ground_truth = {}
with open(DATA_FILE, "r", encoding="utf-8") as gt:
    for number in range(NUM_RECORDS):
        record = json.loads(gt.readline())
        ground_truth[record["id"]] = record
# print(ground_truth)

groups = ["hybrid_conflict", "hybrid_no_conflict", "naive_conflict", "naive_no_conflict"]
scores = {g: {"coverage": [], "semsim": [], "behavior": []} for g in groups}

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
        grp = f"{mode}_conflict" if contradicts else f"{mode}_no_conflict"

        behavior = ConflictBehavior_LLMJudge(question, response)
        print(f"[{mode}] Q{id} Contradicts=[{contradicts}] behavior: {behavior}")
        if grp in scores:
            scores[grp]["behavior"].append(behavior)

        viewpoints = gold.get("answers", [])
        covered = 0
        for item in viewpoints:
            viewpoint = item["answer"]
            is_covered = (BinaryViewpoint_LLMJudge(question, viewpoint, response))
            sim_score = (SemSim_LLMJudge(question, viewpoint, response))
            if is_covered:
                covered += 1
            if grp in scores:
                scores[grp]["semsim"].append(sim_score)
            print(f"[{mode}] Q{id} Contradicts=[{contradicts}] viewpoint: {is_covered} (score={sim_score})")

        if viewpoints and grp in scores:
            scores[grp]["coverage"].append(covered / len(viewpoints))

# Save collected scores to JSON
with open("./data/eval_scores.json", "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2)

print("\nSaved evaluation scores to ./data/eval_scores.json")
print("To plot these scores, run: python plot_eval.py")
