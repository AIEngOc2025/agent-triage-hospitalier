import json
import random

def split_dataset(input_path, train_path, val_path, val_ratio=0.1):
    """
    @definition : Divise le dataset en train/val.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    random.shuffle(data)
    
    val_size = int(len(data) * val_ratio)
    train_data = data[val_size:]
    val_data = data[:val_size]
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(val_path, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ Partitionnement terminé : {len(train_data)} train, {len(val_data)} val.")

if __name__ == "__main__":
    split_dataset(
        "data/processed/train_sft_final_5k_triage.jsonl",
        "data/processed/train_sft_5k.jsonl",
        "data/processed/val_sft_500.jsonl"
    )
