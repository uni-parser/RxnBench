import json
import jsonlines
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import traceback
from collections import defaultdict
import os
from pathlib import Path
from openai import OpenAI

# ================= settings =================
# Model name for API calls and file naming
MODEL_NAME = "gpt-4o" 
# Get API Key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY") 
# Get Base URL from environment variable (optional)
BASE_URL = os.getenv("OPENAI_BASE_URL")
# Output directory: defaults to './results/{MODEL_NAME}'
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./results")) / MODEL_NAME
# Base path (retained from user structure, not directly used here)
BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "RxnBench/pdfqa" 
# ===========================================

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File path definitions
RAW_ANSWERS_PATH = f"/path/to/eval/results/{MODEL_NAME}.json" 
DATASET_PATH = "/path/to/dataset.en.jsonl"

# Output file paths
OUTPUT_JSONL = OUTPUT_DIR / f"{MODEL_NAME}_extracted.jsonl"
ERROR_JSONL = OUTPUT_DIR / f"{MODEL_NAME}_error.jsonl"
ACCURACY_JSON  = OUTPUT_DIR / f"{MODEL_NAME}_metrics.json"

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL # Uses default if BASE_URL is None
)


def run_api(prompt):
    """Call the standard OpenAI Chat Completion API."""
    content = [{"type": "text", "text": prompt}]
    completion = client.chat.completions.create(
        model=MODEL_NAME, # Use model name from settings
        messages=[{"role": "user", "content": content}]
    )
    return completion.choices[0].message.content.strip()


def get_answer(prompt, max_retries=5, delay=1):
    """Call GPT API with exponential backoff for retries."""
    for attempt in range(max_retries):
        try:
            answer = run_api(prompt)
            return answer
        except Exception as e:
            print(f"API Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2**attempt))
                continue
            else:
                raise e


def extract_answer_with_llm(text, max_retries=3):
    """Extract answer using LLM when heuristic extraction fails."""
    prompt = f"""
你是一个严格的答案抽取器。给你一段 GPT 输出内容，其中包含不定项选择题的最终答案，以及可能存在的分析、解释、推理过程、额外文字。你的任务是：

1. 从文本中抽取真正的最终选择题答案。
2. 答案必须只由大写字母 A–E 组成（例如：A、B、C、AB、ACD、E）。
3. 按字母顺序排序（例如：BCA → ABC）。
4. 只输出最终字母序列，不要输出任何解释、不加标点、不换行、不加空格、不加前后文本。
5. 若检测到多个出现“答案”或“correct answer”字段，以最后一个明确答案为准。
6. 若文本出现“None of the above”“无正确答案”“没有正确选项”等语句，则输出选项 E。

下面是需要抽取的文本：
{text}
"""
    for _ in range(max_retries):
        try:
            ans = get_answer(prompt).strip()
            letters = "".join(sorted(set([c for c in ans if c in "ABCDE"])))
            if letters == "":
                # If LLM returns nothing, default to E
                letters = "E" 
            return letters
        except Exception:
            time.sleep(0.2)
    # If all retries fail, default to E
    return "E"


def heuristic_extract(raw):
    """Simple heuristic extraction: gets unique uppercase A-E characters from the first line."""
    # Get the first line
    s = raw.strip().split("\n")[0].strip()

    # Clean the string, keeping only uppercase letters
    s_clean = re.sub(r"[ \t\n\r!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", "", s).upper()

    if not s_clean:
        return None

    # Return None if non A-E letters are found
    if any(c not in "ABCDE" for c in s_clean):
        return None

    # Deduplicate, sort, and return
    return "".join(sorted(set(s_clean)))


def process_item(idx, raw_answer, dataset_item):
    """Process a single evaluation item and extract the predicted answer."""
    try:
        # Normalize gt answer
        gt = dataset_item.get("gt", "").strip()
        gt = "".join(sorted([c for c in gt if c in "ABCDE"]))
        question_type = dataset_item.get("question_type", "Unknown")

        # 1. Fast heuristic extraction
        fast = heuristic_extract(raw_answer)
        if fast is not None:
            pred = fast
        else:
            # 2. LLM assisted extraction
            pred = extract_answer_with_llm(raw_answer)

        return {
            "index": idx,
            "pred": pred,
            "gt": gt,
            "correct": pred == gt,
            "raw_answer": raw_answer,
            "question_type": question_type,
            "question": dataset_item.get("question", "")
        }, None
    except Exception as e:
        return None, {
            "index": idx,
            "error": str(e),
            "trace": traceback.format_exc(),
            "raw_answer": raw_answer,
            "question_type": dataset_item.get("question_type", "Unknown"),
            "question": dataset_item.get("question", "")
        }

# ==============================
# Compute accuracy by question_type
# ==============================
def compute_accuracy(results):
    """Computes overall and per-question-type accuracy."""
    # Tally stats by question_type
    class_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    total = 0
    correct = 0

    for r in results:
        pred = r["pred"]
        gt = r["gt"]
        qtype = r.get("question_type", "Unknown")

        total += 1
        if pred == gt:
            correct += 1

        class_stats[qtype]["total"] += 1
        if pred == gt:
            class_stats[qtype]["correct"] += 1

    # Format results into JSON structure
    stats_json = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0
        },
        "per_question_type": {}
    }

    for qtype, v in class_stats.items():
        t = v["total"]
        crr = v["correct"]
        stats_json["per_question_type"][qtype] = {
            "total": t,
            "correct": crr,
            "accuracy": crr / t if t > 0 else 0
        }

    return stats_json


def main():
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set or API_KEY is empty. Please check the configuration.")
        return

    print("Loading raw answers...")
    try:
        with open(RAW_ANSWERS_PATH, 'r', encoding='utf-8') as f:
            raw_answers = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Raw answers file not found at {RAW_ANSWERS_PATH}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {RAW_ANSWERS_PATH}")
        return

    print("Loading dataset...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {DATASET_PATH}")
        return

    assert len(raw_answers) == len(dataset), \
        f"Count mismatch: {len(raw_answers)} raw answers vs {len(dataset)} dataset items."

    n = len(raw_answers)
    results = []
    errors = []

    print("Processing with multithreading...")
    # Set max workers for concurrency (e.g., based on API limits)
    MAX_WORKERS = 16 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_item, i, raw_answers[i], dataset[i]): i
            for i in range(n)
        }

        for future in tqdm(as_completed(futures), total=n, desc="Extracting Answers"):
            r, err = future.result()
            if r:
                results.append(r)
            if err:
                errors.append(err)

    # Write results
    print(f"Writing results -> {OUTPUT_JSONL}")
    with jsonlines.open(OUTPUT_JSONL, "w") as writer:
        for r in sorted(results, key=lambda x: x["index"]):
            writer.write(r)

    # Write error samples
    if errors:
        print(f"Writing errors -> {ERROR_JSONL}")
        with jsonlines.open(ERROR_JSONL, "w") as writer:
            for e in errors:
                writer.write(e)
    else:
        print("No errors logged.")

    # Compute accuracy
    if not results:
        print("No results to compute accuracy.")
        return
        
    print("Computing accuracy...")
    acc_stats = compute_accuracy(results)

    with open(ACCURACY_JSON, "w", encoding='utf-8') as f:
        json.dump(acc_stats, f, indent=2, ensure_ascii=False)

    print(f"Accuracy saved -> {ACCURACY_JSON}")
    print(f"Overall Accuracy: {acc_stats['overall']['accuracy']:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()
