import csv
import json
import subprocess
import time
from pathlib import Path

RESULTS_CSV = Path("experiments.csv")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

lr_values = [1e-2, 3e-3, 1e-3]
eps_values = [1e-3, 3e-4, 1e-4]

experiments = [
    #{"name": "bn_spsa_balanced_mmnt_xav_64_128_s42", "batch_size": 32, "n_batches": 256, "seed": 42},
    {"name": "bneval_bn_spsa_balanced_mmnt_xav_64_128_s42", "batch_size": 64, "n_batches": 128, "seed": 42},
    #{"name": "bn_spsa_balanced_mmnt_xav_64_128_s42", "batch_size": 16, "n_batches": 512, "seed": 42},
]

fieldnames = [
    "name",
    "seed",
    "batch_size",
    "n_batches",
    "time_sec",
    "imagenet_head_acc",
    "init_head_acc",
    "finetuned_acc",
    "finetuned_acc_percent",
    "output_json",
    "returncode",
]

file_exists = RESULTS_CSV.exists()

with open(RESULTS_CSV, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()

    for exp in experiments:
        output_path = RUNS_DIR / f"{exp['name']}.json"

        cmd = [
            "python",
            "validate.py",
            "--batch_size", str(exp["batch_size"]),
            "--n_batches", str(exp["n_batches"]),
            "--seed", str(exp["seed"]),
            "--output", str(output_path),
        ]

        print(f"\n=== Running {exp['name']} ===")
        start = time.time()

        result = subprocess.run(cmd, text=True)

        elapsed = round(time.time() - start, 2)

        row = {
            "name": exp["name"],
            "seed": exp["seed"],
            "batch_size": exp["batch_size"],
            "n_batches": exp["n_batches"],
            "time_sec": elapsed,
            "imagenet_head_acc": "",
            "init_head_acc": "",
            "finetuned_acc": "",
            "finetuned_acc_percent": "",
            "output_json": str(output_path),
            "returncode": result.returncode,
        }

        if result.returncode == 0 and output_path.exists():
            with open(output_path) as jf:
                metrics = json.load(jf)

            ft = metrics["val_accuracy_top1_finetuned"]

            row.update({
                "imagenet_head_acc": metrics["val_accuracy_top1_imagenet_head"],
                "init_head_acc": metrics["val_accuracy_top1_init_head"],
                "finetuned_acc": ft,
                "finetuned_acc_percent": round(ft * 100, 2),
            })

            print(f"FINETUNED ACC: {ft:.4f} ({ft * 100:.2f}%)")
        else:
            print("Run failed")

        writer.writerow(row)
        f.flush()