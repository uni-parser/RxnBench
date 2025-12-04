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

# ================= settings =================
MODEL_NAME = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./results"))
BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "RxnBench/src_pdf" 
# ===========================================

if not API_KEY:
    print("Warning: OPENAI_API_KEY is not set. The client will rely on standard environment defaults.")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


INPUT_DIR = BASE_PATH / "translated_gpt51_sample_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
            if attempt + 1 == max_retry:
                return ""
            time.sleep(2 * (attempt + 1)) 

    return ""


# ==============================
# Single Item Processing
# ==============================

def process_single_item(args: tuple[int, dict]) -> tuple[int, str]:
    
    index, item = args

    pdf_relative_path = item["pdf_path"]
    question = item["question"]
    image_paths = item["images"] 

    try:
        # 1) Determine full path for PDF
        pdf_full_path = BASE_PATH / pdf_relative_path
        
        # 2) Render PDF to images, or fallback to text extraction
        pdf_images_b64 = render_pdf_to_images_b64(pdf_full_path)
        pdf_component = pdf_images_b64 if pdf_images_b64 else extract_pdf_text(pdf_full_path)

        # 3) Load option images
        option_images_b64 = [load_image_as_base64(p) for p in image_paths]

        # 4) Construct multimodal messages
        messages = build_multimodal_messages(
            question=question,
            option_images_b64=option_images_b64,
            pdf_component=pdf_component
        )

        # 5) Call GPT with retry
        output = call_model_with_retry(messages)

    except Exception:
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
# Main Execution
# ==============================

if __name__ == "__main__":
    
    # --- Example 1: Chinese Dataset (rxnbench_doc.zh.jsonl) ---
    zh_filename = "rxnbench_doc.zh.jsonl"
    zh_input_path = INPUT_DIR / zh_filename
    print(f"Starting Chinese dataset processing: {zh_input_path}")
    
    if zh_input_path.exists():
        zh_res = process_jsonl_parallel(
            zh_input_path,
            num_workers=8
        )
        zh_output_path = OUTPUT_DIR / f"{MODEL_NAME}_zh_doc.json"
        with open(zh_output_path, "w", encoding="utf-8") as f:
            json.dump(zh_res, f, indent=4, ensure_ascii=False)
        print(f"Chinese results saved to: {zh_output_path}")
    else:
        print(f"Warning: Chinese input file not found at {zh_input_path}")
    
    print("-" * 30)

    # --- Example 2: English Dataset (rxnbench_doc.en.jsonl) ---
    en_filename = "rxnbench_doc.en.jsonl"
    en_input_path = INPUT_DIR / en_filename
    print(f"Starting English dataset processing: {en_input_path}")
    
    if en_input_path.exists():
        en_res = process_jsonl_parallel(
            en_input_path,
            num_workers=8
        )
        en_output_path = OUTPUT_DIR / f"{MODEL_NAME}_en_doc.json"
        with open(en_output_path, "w", encoding="utf-8") as f:
            json.dump(en_res, f, indent=4, ensure_ascii=False)
        print(f"English results saved to: {en_output_path}")
    else:
        print(f"Warning: English input file not found at {en_input_path}")
