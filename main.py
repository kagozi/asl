from pathlib import Path
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader


from process_asl import (
    load_aslg_pc12_df,
    preprocess_aslg,
    make_splits,
    save_splits,
    load_saved_splits,
    TranslationDataset,
    collate_batch,
)
from vocab import build_word_vocab, Vocab
from train import train_one_run, RunConfig
from baseline import BaselineTransformer
from modern import ModernTransformer

# -----------------------------
# 0) Config
# -----------------------------
DATA_DIR = Path("data/aslg_pc12")
REFRESH_DATA = False

D_MODEL = 512
NHEAD = 8
DROPOUT = 0.1

BATCH_SIZE = 32
GRAD_ACCUM = 32
EPOCHS = 100          
WARMUP_STEPS = 4000
LR_FACTOR = 1.0

NUM_KV_HEADS = 4
FFN_MULT = 2.7
MAX_DECODE_LEN = 100

SEED = 42

OUT_DIR = "'../santosh_lab/shared/KagoziA/asl/runs"
RESULTS_CSV = Path("../santosh_lab/shared/KagoziA/asl/runs/results.csv")
SKIP_IF_EXISTS_IN_CSV = True  # set False to force rerun


NUM_WORKERS = 2

def size_to_layers(size: str):
    if size == "small":
        return 2, 2
    if size == "medium":
        return 4, 4
    return 6, 6

# -----------------------------
# 1) Load + cache splits once
# -----------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

if REFRESH_DATA or not (DATA_DIR / "train.parquet").exists():
    df = preprocess_aslg(load_aslg_pc12_df())
    splits = make_splits(df)
    save_splits(splits, str(DATA_DIR))
else:
    splits = load_saved_splits(str(DATA_DIR))

# -----------------------------
# 2) Build vocabs once (TRAIN only)
# -----------------------------
text_vocab = build_word_vocab(splits.train["processed_text"], specials=["<pad>", "<unk>"])
gloss_vocab = build_word_vocab(splits.train["processed_gloss"], specials=["<pad>", "<unk>", "<start>", "<end>"])

# -----------------------------
# 3) Create datasets/loaders once
# -----------------------------
train_ds = TranslationDataset(splits.train, text_vocab, gloss_vocab)
val_ds   = TranslationDataset(splits.val, text_vocab, gloss_vocab)
test_ds  = TranslationDataset(splits.test, text_vocab, gloss_vocab)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=lambda b: collate_batch(b, text_vocab, gloss_vocab),
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=lambda b: collate_batch(b, text_vocab, gloss_vocab),
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=lambda b: collate_batch(b, text_vocab, gloss_vocab),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# 4) Helper: check if run already exists
# -----------------------------
def run_already_logged(run_name: str) -> bool:
    if not RESULTS_CSV.exists():
        return False
    df = pd.read_csv(RESULTS_CSV)
    # We log final_test rows; only skip if final_test exists for that run_name
    return ((df["run_name"] == run_name) & (df["epoch"].astype(str) == "final_test")).any()

# -----------------------------
# 5) Sweep
# -----------------------------
MODEL_TYPES = ["baseline", "modern"]
SIZES = ["small", "medium", "large"]

ckpts = {}

for MODEL_TYPE in MODEL_TYPES:
    for SIZE in SIZES:
        enc_layers, dec_layers = size_to_layers(SIZE)
        run_name = f"aslg_{MODEL_TYPE}_{SIZE}_d{D_MODEL}_h{NHEAD}_seed{SEED}"

        if SKIP_IF_EXISTS_IN_CSV and run_already_logged(run_name):
            print(f"⏭️  Skip (already in CSV): {run_name}")
            continue

        # reproducibility per run
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # build model
        if MODEL_TYPE == "baseline":
            model = BaselineTransformer(
                src_vocab_size=len(text_vocab.tokens),
                tgt_vocab_size=len(gloss_vocab.tokens),
                d_model=D_MODEL,
                nhead=NHEAD,
                num_encoder_layers=enc_layers,
                num_decoder_layers=dec_layers,
                dim_feedforward=D_MODEL * 4,
                dropout=DROPOUT,
                pad_id_src=text_vocab.pad_id,
                pad_id_tgt=gloss_vocab.pad_id,
            )
        else:
            model = ModernTransformer(
                src_vocab_size=len(text_vocab.tokens),
                tgt_vocab_size=len(gloss_vocab.tokens),
                d_model=D_MODEL,
                nhead=NHEAD,
                num_encoder_layers=enc_layers,
                num_decoder_layers=dec_layers,
                dropout=DROPOUT,
                num_kv_heads=NUM_KV_HEADS,
                ffn_mult=FFN_MULT,
                pad_id_src=text_vocab.pad_id,
                pad_id_tgt=gloss_vocab.pad_id,
                use_rope=True,
            )

        cfg = RunConfig(
            run_name=run_name,
            model_type=MODEL_TYPE,
            size=SIZE,
            d_model=D_MODEL,
            nhead=NHEAD,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            dropout=DROPOUT,
            num_kv_heads=NUM_KV_HEADS,
            ffn_mult=FFN_MULT,
            lr_factor=LR_FACTOR,
            warmup_steps=WARMUP_STEPS,
            batch_size=BATCH_SIZE,
            grad_accum=GRAD_ACCUM,
            max_tgt_len=0,
            max_decode_len=MAX_DECODE_LEN,
            epochs=EPOCHS,
            seed=SEED,
        )

        print(f"\n🚀 Running: {run_name}")
        ckpt = train_one_run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            text_vocab=text_vocab,
            gloss_vocab=gloss_vocab,
            cfg=cfg,
            device=device,
            out_dir=OUT_DIR,
            results_csv=str(RESULTS_CSV),
        )
        ckpts[run_name] = ckpt
        print(f"✅ Done: {run_name}  | best ckpt: {ckpt}")

print("\nFinished sweep. Checkpoints created:")
for k,v in ckpts.items():
    print(" -", k, "=>", v)
print("\nResults CSV:", RESULTS_CSV)
