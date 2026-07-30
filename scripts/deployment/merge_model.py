import traceback

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use the base model that matches the SFT/DPO training
base_model_id = "Qwen/Qwen3-1.7B-Base"
adapter_path = "models/dpo_final_chsa"
save_path = "models/merged_dpo_final_chsa"


def merge_and_save_model():
    """
    @definition : Merges the PEFT adapters with the base model and saves the
                  resulting model.
    @args/params : None.
    @return : None. Prints success or failure message.
    """
    print("🧬 Fusion du modèle pour déploiement vLLM (Mode CPU/Linux)...")

    try:
        print("📦 [STEP 1/4] Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        print(
            "📦 [STEP 2/4] Chargement du modèle de base (float32 pour stabilité CPU)..."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float32, device_map="auto"
        )

        print("📦 [STEP 3/4] Chargement des adapters et fusion...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()

        print("📦 [STEP 4/4] Sauvegarde du modèle fusionné...")
        merged_model.save_pretrained(save_path, max_shard_size="2GB")
        tokenizer.save_pretrained(save_path)

        print(f"✅ Modèle fusionné et sauvegardé avec succès dans {save_path}")

    except Exception as e:
        print(f"❌ Erreur critique lors de la fusion : {str(e)}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    merge_and_save_model()
