import json
import fitz
import base64
import traceback
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import time
import os
import sys
from openai import OpenAI
from datasets import load_dataset

# ================= settings =================
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
INFER_OUTPUT_DIR = Path(os.getenv("INFER_OUTPUT_DIR", "./results"))
BASE_PATH = Path(os.getenv("BASE_PATH"))
# ===========================================

if not API_KEY:
    print("Warning: OPENAI_API_KEY is not set. The client will rely on standard environment defaults.")

print(f"using {MODEL_NAME=} with {API_KEY=} and {BASE_URL=}, {INFER_OUTPUT_DIR=} and {BASE_PATH=}")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


INPUT_DIR = BASE_PATH
INFER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# PDF Processing Utilities
# ==============================

def render_pdf_to_images_b64(pdf_path: Path, dpi: int = 144) -> list[str] | None:
    try:
        doc = fitz.open(str(pdf_path))
        images = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode()
            images.append(b64)
        return images
    except Exception:
        return None


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        text = [page.get_text() for page in doc]
        return "\n".join(t.strip() for t in text if t.strip())
    except Exception:
        return ""


def load_image_as_base64(relative_path: str) -> str:
    full_path = BASE_PATH / relative_path
    if not full_path.exists():
        print(f"Warning: Image not found: {full_path}. Please prepare the image files as described in the README.md. (can be downloaded from https://huggingface.co/datasets/UniParser/RxnBench-Doc/resolve/main/images.zip)")
        raise FileNotFoundError(f"Image not found: {full_path}")
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ==============================
# OpenAI Message Construction
# ==============================

def build_multimodal_messages(question: str, option_images_b64: list[str], pdf_component: list[str] | str) -> list[dict]:
    
    parts = question.split("<image>")
    content = []

    # 1. Interleave question text parts and option images
    for idx, p in enumerate(parts):
        if p.strip():
            content.append({"type": "text", "text": p.strip()})

        if idx < len(option_images_b64):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{option_images_b64[idx]}"}
            })

    # 2. Append PDF content (image pages or text fallback)
    if isinstance(pdf_component, list):
        for b64 in pdf_component:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
    else:
        content.append({"type": "text", "text": pdf_component})

    return content


# ==============================
# GPT Model Call
# ==============================

def call_model_with_retry(content: list[dict], max_retry: int = 5) -> str:
    
    messages = [{"role": "user", "content": content}]

    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Error calling model: {e}, attempt {attempt + 1} of {max_retry}")
            if attempt + 1 == max_retry:
                print(f"LLM completion max retry reached, returning empty string")
                return ""
            time.sleep(2 * (attempt + 1)) 

    return ""


# ==============================
# Single Item Processing
# ==============================

def process_single_item(args: tuple[int, dict]) -> tuple[int, str]:

    index, item = args

    question = item["question"]
    image_paths = item["images"]

    try:
        # Check if PDF files are available (they might not be due to legal constraints)
        pdf_doi = item.get("pdf_doi", "")
        pdf_available = False

        if pdf_doi and BASE_PATH:
            parsed_pdf_path = BASE_PATH / "pdf_files" / (pdf_doi.replace("/", "_") + ".pdf")
            pdf_available = parsed_pdf_path.exists()

        if pdf_available:
            # PDF is available, use full multimodal approach
            pdf_full_path = parsed_pdf_path
            pdf_images_b64 = render_pdf_to_images_b64(pdf_full_path)
            pdf_component = pdf_images_b64 if pdf_images_b64 else extract_pdf_text(pdf_full_path)
        else:
            # PDF not available, use text-only or alternative approach
            expected_path = BASE_PATH / "pdf_files" / (pdf_doi.replace("/", "_") + ".pdf") if pdf_doi and BASE_PATH else "N/A"
            print(f"Warning: PDF not available for item {index} (DOI: {pdf_doi or 'N/A'}, expected path: {expected_path}). Please prepare PDF files as described in the README.")
            raise FileNotFoundError(f"PDF not available for item {index} (DOI: {pdf_doi or 'N/A'}, expected path: {expected_path})")

        # Load option images
        option_images_b64 = [load_image_as_base64(p) for p in image_paths]

        # Construct multimodal messages
        messages = build_multimodal_messages(
            question=question,
            option_images_b64=option_images_b64,
            pdf_component=pdf_component
        )

        # Call GPT with retry
        output = call_model_with_retry(messages)

    except Exception as e:
        print(f"Error processing item {index}: {e}")
        output = ""

    return index, output


# ==============================
# Parallel JSONL Processing
# ==============================

def process_jsonl_parallel(input_jsonl_path: Path, num_workers: int = 4) -> list[str]:
    
    data = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    tasks = [(i, data[i]) for i in range(len(data))]

    results = [None] * len(data)

    with Pool(num_workers) as pool:
        for idx, output in tqdm(
            pool.imap_unordered(process_single_item, tasks),
            total=len(data),
            desc=f"Processing {input_jsonl_path.name}"
        ):
            results[idx] = output

    return results


# ==============================
# Dataset Processing
# ==============================

def process_hf_dataset(dataset_split, lang="en", num_workers=8):
    """Process a HuggingFace dataset split."""
    data = []
    for i, item in enumerate(dataset_split):
        # {'pdf_title': Value(dtype='string', id=None),
        # 'pdf_doi': Value(dtype='string', id=None),
        # 'pdf_link': Value(dtype='string', id=None),
        # 'question': Value(dtype='string', id=None),
        # 'images': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        # 'gt': Value(dtype='string', id=None),
        # 'reasoning': Value(dtype='string', id=None),
        # 'question_type': Value(dtype='string', id=None)}
        processed_item = {
            "pdf_title": item.get("pdf_title", ""),
            "pdf_doi": item.get("pdf_doi", ""),
            "pdf_link": item.get("pdf_link", ""),
            "question": item.get("question", ""),
            "images": item.get("images", []),
            "gt": item.get("gt", ""),
            "reasoning": item.get("reasoning", ""),
            "question_type": item.get("question_type", ""),
        }
        data.append(processed_item)

    print(f"Processing {lang} split with {len(data)} samples...")

    tasks = [(i, data[i]) for i in range(len(data))]
    results = [None] * len(data)

    with Pool(num_workers) as pool:
        for idx, output in tqdm(
            pool.imap_unordered(process_single_item, tasks),
            total=len(data),
            desc=f"Processing {lang} split"
        ):
            results[idx] = output

    return results


# ==============================
# Main Execution
# ==============================

def main():
    try:
        ds = load_dataset(
            "UniParser/RxnBench-Doc",
            data_files={
                "zh": "rxnbench_doc.zh.jsonl",
                "en": "rxnbench_doc.en.jsonl",
            }
        )
        print("Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")
        for split_name in ds.keys():
            print(f"  - {split_name}: {len(ds[split_name])} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have access to UniParser/RxnBench-Doc on HuggingFace")
        return

    print()

    # Process each language split
    for lang in ["en", "zh"]:
        if lang in ds:
            print(f"Processing {lang} split...")
            results = process_hf_dataset(ds[lang], lang=lang, num_workers=8)

            output_path = INFER_OUTPUT_DIR / f"{MODEL_NAME}_{lang}_doc.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
            print("-" * 50)
        else:
            print(f"Warning: {lang} split not found in dataset")

if __name__ == "__main__":
    main()
