import json
import requests
import time

SERVER = "http://localhost:9621" # your running lightrag server
DATA_FILE = "./data/ConfRAG.jsonl" # make sure dataset is downloaded here successfuly
NUM_RECORDS = 1  # 1 record can have up to 10 website scrapes

'CLEAR CURRENT DATA IF REQUESTED'
q = input("Clear current data before loading? (y/n): ").strip().lower()
if q == "y":
    del_resp = requests.delete(f"{SERVER}/documents", timeout=512)
    print(f"Server Response: {del_resp.status_code}")

with open(DATA_FILE, "r", encoding="utf-8") as file:
    num = 0
    for i in range(NUM_RECORDS):
        record = json.loads(file.readline())
        websites = record["websites"]
        
        print(f"There are {len(websites)} in this record")
        
        for entry in websites:
            content = entry["content"]
            if not content:
                continue
            resp = requests.post(
                f"{SERVER}/documents/text",
                json={"text": content, "file_source": str(num)},
                timeout=512
            )
            num += 1
            print(content)
            
            print(f"Detailed Response: {resp.text}")
            print(f"Server Response: ({resp.status_code})")

# print("Wait for success...")
# time.sleep(20)
# for i in range(1, 3):
#     docs = requests.get(f"{SERVER}/documents", timeout=2000).json()
#     failed = docs.get("statuses", {}).get("FAILED", [])

#     if not failed:
#         print("Success!")
#         break
#     requests.post(f"{SERVER}/documents/scan", timeout=30)
#     time.sleep(10)