import json
import httpx
import asyncio
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/chat"
INPUT_FILE = "data/processed/train_sft_final_5k_triage.jsonl"
OUTPUT_FILE = "data/processed/labeled_triage.jsonl"
SAMPLE_SIZE = 100

async def label_data():
    if not Path(INPUT_FILE).exists():
        print(f"❌ Fichier non trouvé : {INPUT_FILE}")
        return

    # Utilisation d'un client HTTP asynchrone
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
            count = 0
            for line in f_in:
                if count >= SAMPLE_SIZE: break
                
                try:
                    data = json.loads(line)
                    instruction = data.get("instruction", "")
                    
                    # Appel à l'API pour obtenir le triage
                    response = await client.post(
                        API_URL, 
                        json={"history": [{"role": "user", "content": instruction}], "stream": False}
                    )
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        # Vérification de la structure attendue
                        if "triage" in res_json:
                            # Sauvegarde du texte et du label
                            labeled_data = {
                                "text": instruction,
                                "label": res_json["triage"]["niveau"]
                            }
                            f_out.write(json.dumps(labeled_data) + "\n")
                            count += 1
                            print(f"✅ Étiqueté {count}/{SAMPLE_SIZE}: {res_json['triage']['niveau']}")
                        else:
                            print(f"⚠️ Pas de triage pour: {instruction[:50]}...")
                
                except Exception as e:
                    print(f"❌ Erreur sur ligne {count}: {e}")

if __name__ == "__main__":
    asyncio.run(label_data())
    print(f"🎉 Dataset étiqueté sauvegardé dans {OUTPUT_FILE}")
