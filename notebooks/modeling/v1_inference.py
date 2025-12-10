# ===============================================================
# ensemble_inference.py — Seed별 T5 요약 + Ensemble + 제출 파일 생성 (fname 포함)
# ===============================================================

import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ config ------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


# ------------------------------ seed dirs ------------------------------
def find_seed_dirs(base_output_dir):
    if not os.path.exists(base_output_dir):
        raise FileNotFoundError(f"❌ output_dir 없음: {base_output_dir}")

    seed_dirs = [
        os.path.join(base_output_dir, d)
        for d in os.listdir(base_output_dir)
        if d.startswith("seed_") and os.path.isdir(os.path.join(base_output_dir, d))
    ]

    if not seed_dirs:
        print(f"⚠️ seed_* 없음 → 단일 모델로 처리: {base_output_dir}")
        return [base_output_dir]

    print(f"✅ 발견된 seed 디렉토리: {seed_dirs}")
    return seed_dirs


def get_best_checkpoint_path_in_dir(model_dir):
    subdirs = [
        d for d in os.listdir(model_dir)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(model_dir, d))
    ]

    if not subdirs:
        print(f"📌 [{model_dir}] checkpoint-* 없음 → 본 디렉토리 사용")
        return model_dir

    subdirs = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))
    best = os.path.join(model_dir, subdirs[-1])
    print(f"📌 [{model_dir}] best checkpoint 선택: {best}")
    return best


# ------------------------------ model load ------------------------------
def load_model_and_tokenizer_for_checkpoint(cfg, checkpoint_path):
    tokenizer = T5TokenizerFast.from_pretrained(cfg["general"]["model_name"])
    tokenizer.add_tokens(cfg["tokenizer"]["special_tokens"])

    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ------------------------------ test load ------------------------------
def load_test_dataset(cfg):
    test_path = os.path.join(cfg["general"]["data_dir"], cfg["general"]["test_file"])
    df = pd.read_csv(test_path)

    df = df.rename(columns={"dialogue_clean": "input_text"})
    print(f"📌 Test size: {len(df)} rows")
    return df


# ------------------------------ generate ------------------------------
def clean_summary(text, remove_tokens):
    for t in remove_tokens:
        text = text.replace(t, "")
    return text.strip()


def generate_summaries_for_model(model, tokenizer, df, cfg):
    results = []

    batch_size = cfg["inference"]["batch_size"]
    num_beams = cfg["inference"]["num_beams"]
    max_len = cfg["inference"]["max_length"]
    no_repeat = cfg["inference"]["no_repeat_ngram_size"]
    prefix = cfg["general"].get("prefix", "")
    remove_tokens = cfg["inference"]["remove_tokens"]

    for i in tqdm(range(0, len(df), batch_size), desc="Generating"):
        batch_text = [prefix + x for x in df["input_text"].iloc[i:i+batch_size].tolist()]

        inputs = tokenizer(
            batch_text,
            max_length=cfg["tokenizer"]["encoder_max_len"],
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [clean_summary(t, remove_tokens) for t in decoded]
        results.extend(cleaned)

    return results


# ------------------------------ main ------------------------------
def main():
    test_df = load_test_dataset(config)

    output_dir = config["general"]["output_dir"]
    seed_dirs = find_seed_dirs(output_dir)

    all_outputs = []

    for sd in seed_dirs:
        ckpt = get_best_checkpoint_path_in_dir(sd)

        tokenizer, model = load_model_and_tokenizer_for_checkpoint(config, ckpt)

        preds = generate_summaries_for_model(model, tokenizer, test_df, config)

        fname = sd.split("/")[-1]
        out_path = f"submission_{fname}.csv"
        pd.DataFrame({"fname": test_df["fname"], "summary": preds}).to_csv(out_path, index=False)
        print(f"📁 Saved → {out_path}")

        all_outputs.append(preds)

    # ensemble (simple average by length closeness)
    if len(all_outputs) > 1:
        final = []
        for i in range(len(test_df)):
            cands = [out[i] for out in all_outputs]
            best = min(cands, key=lambda x: abs(len(x) - 40))
            final.append(best)

        pd.DataFrame({"fname": test_df["fname"], "summary": final}).to_csv(
            "submission_ensemble.csv", index=False
        )
        print("📁 Saved → submission_ensemble.csv")


if __name__ == "__main__":
    main()
