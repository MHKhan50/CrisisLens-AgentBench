import argparse
import json
from pathlib import Path
from .data import load_training_csv, load_reports_jsonl
from .modeling import train_model, load_model, evaluate_model
from .pipeline import CrisisLensPipeline

def build_parser():
    parser = argparse.ArgumentParser(prog="crisislens",
        description="Evidence-preserving humanitarian NLP research benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train the TF-IDF + logistic-regression baseline.")
    train.add_argument("--data", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--seed", type=int, default=17)
    evaluate = sub.add_parser("evaluate", help="Evaluate on the supplied CSV.")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output", default="reports/evaluation.json")
    analyse = sub.add_parser("analyse", help="Run the multi-agent pipeline on JSONL reports.")
    analyse.add_argument("--input", required=True)
    analyse.add_argument("--model", required=True)
    analyse.add_argument("--output", required=True)
    hs = sub.add_parser("humset-download", help="Download HumSet from the Hugging Face Hub (not redistributed).")
    hs.add_argument("--data-dir", default="data/humset")
    au = sub.add_parser("humset-audit", help="Audit HumSet: sizes, languages, labels, duplicates, document leakage.")
    au.add_argument("--data-dir", default="data/humset")
    au.add_argument("--reports", default="reports")
    au.add_argument("--limit", type=int, default=None)
    bl = sub.add_parser("humset-baselines", help="Prior and TF-IDF baselines on every HumSet task.")
    bl.add_argument("--data-dir", default="data/humset")
    bl.add_argument("--reports", default="reports")
    bl.add_argument("--limit", type=int, default=None)
    bl.add_argument("--n-boot", type=int, default=1000)
    bl.add_argument("--tasks", default=",".join(["sectors", "pillars_1d", "pillars_2d", "subpillars_1d", "subpillars_2d"]))
    return parser

def main():
    args = build_parser().parse_args()
    if args.command == "train":
        frame = load_training_csv(args.data)
        model = train_model(frame, args.model, args.seed)
        print("Training complete: {} rows, {} labels.".format(len(frame), len(model.classes_)))
        print("Model saved to {}".format(args.model))
    elif args.command == "evaluate":
        frame = load_training_csv(args.data)
        result = evaluate_model(load_model(args.model), frame)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "classification_report"}, indent=2))
        print("Full evaluation saved to {}".format(path))
    elif args.command == "analyse":
        reports = load_reports_jsonl(args.input)
        pipeline = CrisisLensPipeline(load_model(args.model))
        results = [pipeline.analyse(report) for report in reports]
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Analysed {} reports. Output: {}".format(len(results), path))

    elif args.command == "humset-download":
        from .humset import download
        root = download(args.data_dir)
        print((root / "manifest.json").read_text())
    elif args.command == "humset-audit":
        from .audit import audit, to_markdown
        from .humset import load
        rep = audit(load(args.data_dir, args.limit))
        out = Path(args.reports); out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "audit.md").write_text(to_markdown(rep), encoding="utf-8")
        print(to_markdown(rep))
    elif args.command == "humset-baselines":
        from .audit import norm_text
        from .benchmark import grouped_resplit, run_all, summary_markdown
        from .humset import load
        df = load(args.data_dir, args.limit)
        tasks = args.tasks.split(",")
        out = Path(args.reports); out.mkdir(parents=True, exist_ok=True)
        # Test rows whose document or normalised text also occurs in training are flagged and also scored apart.
        norm = df["excerpt"].map(norm_text)
        train = df["split"] == "train"
        flagged = df["lead_id"].isin(set(df.loc[train, "lead_id"])) | norm.isin(set(norm[train]))
        md = ["# HumSet baselines", ""]
        official = run_all(df, str(out / "runs"), tasks, args.n_boot, exclude_test_rows=flagged, tag="official")
        md.append(summary_markdown(official, "Official split"))
        regrouped = grouped_resplit(df, seed=0)
        grouped = run_all(regrouped, str(out / "runs"), tasks, args.n_boot, tag="grouped")
        md.append(summary_markdown(grouped, "Document-grouped split (no document in two splits)"))
        (out / "baselines.md").write_text("\n".join(md), encoding="utf-8")
        print("\n".join(md))

if __name__ == "__main__":
    main()
