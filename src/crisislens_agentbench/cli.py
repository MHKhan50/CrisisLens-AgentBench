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

if __name__ == "__main__":
    main()
