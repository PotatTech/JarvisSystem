import onnx
import os
import json

MODEL_PATH = "./jarvis/en/en_GB/jarvis/high/jarvis-high.onnx"
CONFIG_PATH = "./jarvis/en/en_GB/jarvis/high/jarvis-high.onnx.json"
OUTPUT_DIR = "./jarvis/en/en_GB/jarvis/high/"

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

model = onnx.load(MODEL_PATH)
meta = model.metadata_props

def add_meta(key, value):
    a = meta.add()
    a.key = key
    a.value = str(value)

# --- Метаданные ---
sample_rate = 22050
if isinstance(config.get("audio"), dict):
    sample_rate = config["audio"].get("sample_rate", 22050)
elif "sample_rate" in config:
    sample_rate = config["sample_rate"]

# ВАЖНО: добавляем "comment" — его не хватало
add_meta("model_type", "vits")
add_meta("comment", "piper")          # ← ФИКС
add_meta("language", "en-GB")
add_meta("voice", "jarvis")
add_meta("has_espeak", 1)
add_meta("n_speakers", 1)
add_meta("sample_rate", sample_rate)

# --- tokens.txt (если ещё нет) ---
tokens_path = os.path.join(OUTPUT_DIR, "tokens.txt")
if not os.path.exists(tokens_path):
    phone_id_map = config.get("phoneme_id_map", {})
    with open(tokens_path, "w", encoding="utf-8") as f:
        for phoneme, ids in phone_id_map.items():
            if phoneme in ("\n", "\r"):
                continue
            token_id = ids[0] if isinstance(ids, list) else ids
            if phoneme == " ":
                phoneme = "▁"
            f.write(f"{phoneme} {token_id}\n")
    print("✅ tokens.txt создан")

onnx.save(model, MODEL_PATH)
print("✅ Метаданные добавлены (включая comment)")