"""Plot matched completed runs, showing mean +/- sample SD across seeds."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/final_5iter"))
    args = parser.parse_args()
    runs, configs, sources = [], [], []
    for path in sorted(args.results_dir.glob("*/config.json")):
        config = json.loads(path.read_text())
        if config["status"] != "complete":
            print(f"Skipping unfinished run: {path.parent}")
            continue
        with (path.parent / "metrics.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != config["iterations"] + 1:
            raise ValueError(f"Incomplete metrics: {path.parent}")
        if [r["method"] for r in rows] != ["no_structure", "structured_bc"] + ["dagger"] * (config["iterations"]-1):
            raise ValueError(f"Unexpected metric order: {path.parent}")
        if [int(r["iteration"]) for r in rows] != list(range(config["iterations"]+1)):
            raise ValueError(f"Unexpected iteration numbers: {path.parent}")
        configs.append(config)
        runs.append([float(r["accuracy"]) for r in rows])
        sources.append(str(path.parent))
    if len(runs) < 2:
        raise ValueError("At least two completed runs are needed for sample SD.")
    first = configs[0]
    for config in configs[1:]:
        for key in ["iterations", "evaluation_fold", "evaluation_role", "train_folds", "beta_decay", "dataset_sha256", "source_sha256", "classifier"]:
            if config[key] != first[key]:
                raise ValueError(f"Runs differ in {key}.")
        params = lambda c: {k:v for k,v in c["classifier_params"].items() if k != "random_state"}
        if params(config) != params(first):
            raise ValueError("Classifier settings differ across runs.")
    seeds = [c["seed"] for c in configs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds: select one matched run per seed.")
    data = np.asarray(runs)
    mean, sd = data.mean(axis=0), data.std(axis=0, ddof=1)
    x = np.arange(1, first["iterations"]+1)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    curves = [("DAgger", mean[1:], sd[1:], "#1565C0", "-"),
              ("Structured BC", np.full(len(x),mean[1]), np.full(len(x),sd[1]), "#D2691E", "--"),
              ("No-structure supervised", np.full(len(x),mean[0]), np.full(len(x),sd[0]), "#21855B", ":")]
    for name, values, spread, color, style in curves:
        ax.plot(x, values, label=name, color=color, linestyle=style,
                marker="o" if name == "DAgger" else None, linewidth=2)
        ax.fill_between(x, values-spread, values+spread, color=color, alpha=0.14)
    for iteration, value in zip(x,mean[1:]):
        ax.annotate(f"{value:.2%}",(iteration,value),xytext=(0,12),textcoords="offset points",ha="center",fontsize=10)
    ax.set(title=f"Stanford OCR: DAgger and supervised baselines (fold {first['evaluation_fold']})",
           xlabel="DAgger iteration",ylabel="Character accuracy",xticks=x)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.margins(x=0.08,y=0.25)
    ax.legend(loc="lower right")
    fig.text(0.5,0.025,f"{first['classifier']} | alpha={first['classifier_params'].get('alpha')} | seeds {', '.join(map(str,seeds))} | bands: +/-1 sample SD across seeds",ha="center",fontsize=9,color="#444444")
    fig.tight_layout(rect=(0,0.055,1,1))
    output = args.results_dir / "combined_accuracy"
    for suffix in [".png", ".pdf"]:
        fig.savefig(output.with_suffix(suffix),bbox_inches="tight")
    plt.close(fig)
    with output.with_suffix(".csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(["method","iteration","mean_accuracy","sample_sd","seeds"])
        for name,values,spread,_,_ in curves:
            for iteration,value,variation in zip(x,values,spread):
                writer.writerow([name,iteration,value,variation,len(seeds)])
    output.with_suffix(".json").write_text(json.dumps({"source_runs":sources,"seeds":seeds,"band":"sample standard deviation across seeds, not a confidence interval"},indent=2))
    print(f"Saved combined plot and summary to {output}.*")


if __name__ == "__main__":
    main()
