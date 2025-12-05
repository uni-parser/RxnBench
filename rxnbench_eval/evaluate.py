import argparse
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ================= settings =================
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.getenv("INFER_OUTPUT_DIR", os.path.join(BASE_DIR, "results"))
ERROR_DIR = os.path.join(RESULT_DIR, "errors")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# ===========================================

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def call_judge(prompt: str, max_retry: int = 3) -> str:
    """Call judge model; return empty string on failure."""
    for i in range(max_retry):
        try:
            return (
                client.chat.completions.create(
                    model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
                )
                .choices[0]
                .message.content.strip()
            )
        except Exception as e:
            logger.warning(f"Judge API attempt {i + 1}: {e}")
            if i < max_retry - 1:
                time.sleep(1)
    logger.error("Judge API failed")
    return ""


def extract_answer(choices: List[str], text: str) -> int:
    """Map model text → 0/1/2/3; return -1 if fail."""
    if not text:
        return -1
    text = text.strip()

    # 1. single letter
    if len(text) == 1 and text.upper() in "ABCD":
        return ord(text.upper()) - ord("A")

    # 2. last-5-chars heuristic
    if len(text) < 100:
        last = text[-5:].upper()
        hits = [c for c in "ABCD" if c in last]
        if len(hits) == 1:
            return ord(hits[0]) - ord("A")

    # 3. LLM judge
    prompt = (
        f"Select the best option.\nModel: {text}\n"
        f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
        f"Reply only A/B/C/D."
    )
    judge = call_judge(prompt)
    if judge and judge[0].upper() in "ABCD":
        return ord(judge[0].upper()) - ord("A")

    logger.warning(f"Extraction failed: {text[:100]}")
    return -1


def process_single_item(idx: int, data: Dict[str, Any], pred_text: str) -> tuple:
    """Process a single evaluation item and return result and error info."""
    try:
        gt_idx = data["answer"]
        pred_idx = extract_answer(data["choices"], pred_text)
        ok = pred_idx == gt_idx

        result = {
            "idx": idx,
            "gt_idx": gt_idx,
            "pred_idx": pred_idx,
            "ok": ok,
            "question_type": data.get("question_type", "Unknown"),
            "question": data.get("question", ""),
            "choices": data["choices"],
            "pred_raw": pred_text,
        }

        if not ok:
            error = {
                "q": data.get("question", ""),
                "choices": data["choices"],
                "gt": gt_idx,
                "pred_raw": pred_text,
                "pred_idx": pred_idx,
                "type": data.get("question_type", "Unknown"),
            }
        else:
            error = None

        return result, error

    except Exception as e:
        logger.error(f"Error processing item {idx}: {e}")
        error = {
            "idx": idx,
            "error": str(e),
            "question": data.get("question", ""),
            "question_type": data.get("question_type", "Unknown"),
            "pred_raw": pred_text,
        }
        return None, error


def evaluate(model: str, lang: str, dataset) -> float | None:
    """Compute accuracy for one model/lang."""
    pred_file = os.path.join(RESULT_DIR, f"{model}_{lang}.json")
    if not os.path.exists(pred_file):
        logger.error(f"Missing {pred_file}")
        return None

    gt = dataset[lang]
    with open(pred_file, encoding="utf-8") as f:
        preds = json.load(f)

    if len(gt) != len(preds):
        logger.warning("Length mismatch")
        n = min(len(gt), len(preds))
        gt, preds = gt.select(range(n)), preds[:n]

    n = len(gt)
    correct = 0
    errors = []
    stats = defaultdict(lambda: {"c": 0, "t": 0})

    print("Processing with multithreading...")
    # Set max workers for concurrency (e.g., based on API limits)
    MAX_WORKERS = 16
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_item, idx, gt[idx], preds[idx]): idx
            for idx in range(n)
        }

        for future in tqdm(as_completed(futures), total=n, desc=f"{model}-{lang}"):
            result, error = future.result()
            if result:
                ok = result["ok"]
                correct += ok

                t = result["question_type"]
                if t is None:
                    t = "Unknown"
                stats[t]["t"] += 1
                if ok:
                    stats[t]["c"] += 1

            if error:
                errors.append(error)

    acc = correct / n
    logger.info(f"[{model}-{lang}] Acc: {acc:.4f}")

    # print category table
    print("\n" + "-" * 50)
    print(f"{'Category':<30} | {'Acc':<10} | Count")
    print("-" * 50)
    print(f"{stats=}")
    for k in sorted(stats):
        c, t = stats[k]["c"], stats[k]["t"]
        print(f"{k:<30} | {c / t:.4f}     | {c}/{t}")
    print("-" * 50)

    os.makedirs(ERROR_DIR, exist_ok=True)
    with open(
        os.path.join(ERROR_DIR, f"{model}_{lang}_errors.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    return acc


def eval_model(model: str, langs: List[str], dataset) -> dict:
    """Evaluate one model across languages."""
    res = {}
    for lang in langs:
        acc = evaluate(model, lang, dataset)
        if acc is not None:
            res[lang] = acc
    if len(res) == len(langs):
        res["mean"] = sum(res.values()) / len(res)
        logger.info(f">>> {model} mean: {res['mean']:.4f}")
    return res


def main(language="en"):
    logger.info("Evaluation start")
    logger.info(f"Models: {MODEL_NAME}")
    logger.info(f"Language: {language}")
    dataset = load_dataset("UniParser/RxnBench")
    logger.info(f"  {language}: {len(dataset[language])} samples")

    summary = {}
    summary[MODEL_NAME] = eval_model(MODEL_NAME, language, dataset)

    if summary:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for m, r in summary.items():
            if "mean" in r:
                logger.info(f"{m:<30}: {r['mean']:.4f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main(language="en")
    main(language="zh")
