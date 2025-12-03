# ===============================================================
# train.py — Optuna + WandB + ROUGE 평가 + Seed Ensemble (완전 수정본)
# ===============================================================

import os
import yaml
import wandb
import optuna
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from functools import partial
from rouge_score import rouge_scorer
from transformers import set_seed

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

torch.backends.cuda.matmul.allow_tf32 = True


# ===============================================================
# 0) Load config.yaml
# ===============================================================
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


# ===============================================================
# 1) Load dataset
# ===============================================================
def load_train_valid(cfg):
    df_path = os.path.join(cfg["general"]["data_dir"], cfg["general"]["train_file"])
    df = pd.read_csv(df_path)

    df = df[["dialogue_clean", "summary"]].rename(
        columns={"dialogue_clean": "input_text", "summary": "target_text"}
    )

    train_df = df.sample(frac=0.9, random_state=cfg["general"]["seed"])
    valid_df = df.drop(train_df.index)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))

    return train_ds, valid_ds, valid_df.reset_index(drop=True)


train_dataset, valid_dataset, valid_df_pandas = load_train_valid(config)

print("\n===== DEBUG: Dataset Check =====")
print("Train size:", len(train_dataset))
print("Valid size:", len(valid_dataset))
print("=================================\n")


# ===============================================================
# 2) Tokenizer + Model
# ===============================================================
def create_tokenizer_and_model(cfg):
    tokenizer = T5TokenizerFast.from_pretrained(cfg["general"]["model_name"])
    num_added = tokenizer.add_tokens(cfg["tokenizer"]["special_tokens"])

    model = T5ForConditionalGeneration.from_pretrained(cfg["general"]["model_name"])
    model.resize_token_embeddings(len(tokenizer))

    print("\n===== DEBUG: Tokenizer / Model =====")
    print("Special tokens added:", num_added)
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model embedding size:", model.get_input_embeddings().weight.shape[0])
    print("=====================================\n")

    return tokenizer, model


# ===============================================================
# 3) Preprocess
# ===============================================================
def preprocess(batch, tokenizer, cfg_tokenizer, prefix=""):
    inputs = [prefix + x for x in batch["input_text"]]

    enc = tokenizer(
        inputs,
        max_length=cfg_tokenizer["encoder_max_len"],
        truncation=True,
        padding="max_length",
    )

    dec = tokenizer(
        batch["target_text"],
        max_length=cfg_tokenizer["decoder_max_len"],
        truncation=True,
        padding="max_length",
    )["input_ids"]

    pad = tokenizer.pad_token_id
    enc["labels"] = [[t if t != pad else -100 for t in seq] for seq in dec]

    return enc


# ===============================================================
# 4) Tokenization Preview
# ===============================================================
def debug_tokenization_preview(tokenizer, cfg_tokenizer, prefix, dataset):
    sample = dataset[0]
    input_text = prefix + sample["input_text"]
    target_text = sample["target_text"]

    enc = tokenizer(input_text, max_length=cfg_tokenizer["encoder_max_len"], truncation=True)
    dec = tokenizer(target_text, max_length=cfg_tokenizer["decoder_max_len"], truncation=True)

    print("\n===== DEBUG: Tokenization Preview =====")
    print("Input token length:", len(enc["input_ids"]))
    print("Target token length:", len(dec["input_ids"]))
    print("=======================================\n")


# ===============================================================
# 5) ROUGE + compute_metrics
# ===============================================================
def compute_rouge_scores(pred, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1 = np.mean([scorer.score(r, pred)["rouge1"].fmeasure for r in refs])
    r2 = np.mean([scorer.score(r, pred)["rouge2"].fmeasure for r in refs])
    rl = np.mean([scorer.score(r, pred)["rougeL"].fmeasure for r in refs])

    return float(r1 + r2 + rl)


def compute_metrics(eval_pred, tokenizer, gold_df):
    print("⚡ compute_metrics CALLED!")
    preds, labels = eval_pred

    if preds.ndim == 3:
        preds = preds.argmax(-1)

    preds = np.clip(preds.astype(np.int64), 0, tokenizer.vocab_size - 1)

    decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded = [d.strip() for d in decoded]

    scores = []
    for i, pred in enumerate(decoded):
        refs = gold_df.iloc[i]["target_text"].split("|||")
        scores.append(compute_rouge_scores(pred, refs))

    return {"final_rouge": float(np.mean(scores))}


# ===============================================================
# 6) Optuna Trial
# ===============================================================
def run_trial(trial, config, train_dataset, valid_dataset, valid_df):

    lr = float(trial.suggest_categorical("learning_rate", config["optuna"]["search_space"]["learning_rate"]))
    warm = float(trial.suggest_categorical("warmup_ratio", config["optuna"]["search_space"]["warmup_ratio"]))
    epochs = int(trial.suggest_categorical("num_train_epochs", config["optuna"]["search_space"]["num_train_epochs"]))
    bs = config["training"]["per_device_train_batch_size"]

    print(f"\n===== DEBUG: Trial {trial.number} / lr={lr}, warm={warm}, ep={epochs} =====\n")

    tokenizer, model = create_tokenizer_and_model(config)
    prefix = config["general"]["prefix"]

    tok_train = train_dataset.map(
        partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config["tokenizer"], prefix=prefix),
        batched=True, remove_columns=train_dataset.column_names
    )
    tok_valid = valid_dataset.map(
        partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config["tokenizer"], prefix=prefix),
        batched=True, remove_columns=valid_dataset.column_names
    )

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=f"{config['wandb']['name']}_trial{trial.number}",
        config={"lr": lr, "warmup": warm, "epochs": epochs},
        mode=config["wandb"]["mode"],
        reinit=True,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=f"{config['general']['output_dir']}/trial_{trial.number}",
        learning_rate=lr,
        warmup_ratio=warm,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        logging_steps=100,
        predict_with_generate=True,
        save_strategy="epoch",
        save_total_limit=1,
        evaluation_strategy="no",
        fp16=config["training"]["fp16"],
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tok_train)

    trainer.train()
    preds = trainer.predict(tok_valid).predictions

    score = compute_metrics((preds, None), tokenizer, valid_df)["final_rouge"]

    wandb.log({"final_rouge": score})
    wandb.finish()

    del trainer, model, tokenizer, tok_train, tok_valid
    torch.cuda.empty_cache()

    return score


# ===============================================================
# 7) Main — Optuna → Seed Ensemble
# ===============================================================
def main():

    print("\n===== DEBUG: Global GPU =====")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("========================================\n")

    prefix = config["general"]["prefix"]

    # Preview
    temp_tok = T5TokenizerFast.from_pretrained(config["general"]["model_name"])
    debug_tokenization_preview(temp_tok, config["tokenizer"], prefix, train_dataset)
    del temp_tok

    # ---------------------------
    # OPTUNA
    # ---------------------------
    if config["optuna"]["use"]:
        print("🔥 Optuna Search 시작!")

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda tr: run_trial(tr, config, train_dataset, valid_dataset, valid_df_pandas),
            n_trials=config["optuna"]["n_trials"],
        )

        best = study.best_trial.params
        best_lr = float(best["learning_rate"])
        best_warm = float(best["warmup_ratio"])
        best_ep = int(best["num_train_epochs"])

        print("\n🎉 Best Trial:", best)

    else:
        best_lr = config["training"]["learning_rate"]
        best_warm = config["training"]["warmup_ratio"]
        best_ep = config["training"]["num_train_epochs"]

    # ---------------------------
    # SEED ENSEMBLE
    # ---------------------------
    seeds = [42, 2025]
    print(f"\n🔧 Seed Ensemble 시작 — seeds = {seeds}")

    for seed in seeds:
        print(f"\n===== 🚀 Seed {seed} =====")
        set_seed(seed)

        tokenizer, model = create_tokenizer_and_model(config)

        tok_train = train_dataset.map(
            partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config["tokenizer"], prefix=prefix),
            batched=True, remove_columns=train_dataset.column_names
        )
        tok_valid = valid_dataset.map(
            partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config["tokenizer"], prefix=prefix),
            batched=True, remove_columns=valid_dataset.column_names
        )

        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            name=f"{config['wandb']['name']}_seed{seed}",
            config={"lr": best_lr, "warmup": best_warm, "epochs": best_ep, "seed": seed},
            mode=config["wandb"]["mode"],
            reinit=True,
        )

        args = Seq2SeqTrainingArguments(
            output_dir=f"{config['general']['output_dir']}/seed_{seed}",
            learning_rate=best_lr,
            warmup_ratio=best_warm,
            num_train_epochs=best_ep,
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            save_strategy="epoch",
            save_total_limit=2,
            predict_with_generate=True,
            logging_steps=100,
            fp16=config["training"]["fp16"],
            report_to="wandb",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tok_train,
            eval_dataset=tok_valid,
            tokenizer=tokenizer,
            compute_metrics=lambda x: compute_metrics(x, tokenizer, valid_df_pandas),
        )

        trainer.train()
        evals = trainer.evaluate()
        print(f"📊 Seed {seed} Eval:", evals)

        wandb.finish()

        del trainer, model, tokenizer, tok_train, tok_valid
        torch.cuda.empty_cache()

    print("\n🎉 전체 Seed Ensemble 완료!")


if __name__ == "__main__":
    main()
