from huggingface_hub import hf_hub_download


file_path = hf_hub_download(
    repo_id="OracleY/ConfRAG", 
    filename="ConfRAG.jsonl",
    repo_type="dataset",
    local_dir="./data"
)
print(f"File downloaded to: {file_path}")