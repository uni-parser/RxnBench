import os
import time
import json
import base64
from io import BytesIO
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
from datasets import load_dataset


# ================= settings =================
MODEL_NAME = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./results")
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_vqa_prompt(question, choices, lang="en"):
    if isinstance(choices, dict):
        choice_list = list(choices.values())
    else:
        choice_list = choices

    if lang == "zh":
        options_str = "\n".join([f"{chr(65+i)}. {v}" for i, v in enumerate(choice_list)])
        prompt = (
            f"问题: {question}\n"
            f"选项:\n{options_str}\n\n"
            f"请根据图像和问题，从以上四个选项中选择最合适的答案。\n"
            f"只输出单个字母 (A, B, C 或 D)，不要输出选项内容，也不要输出任何解释。"
        )
    else:
        options_str = "\n".join([f"{chr(65+i)}. {v}" for i, v in enumerate(choice_list)])
        prompt = (
            f"Question: {question}\n"
            f"Choices:\n{options_str}\n"
            f"Based on the image and the question, choose the most appropriate answer.\n"
            f"**Only output a single letter (A, B, C, or D)**. Do NOT output any other text or explanation."
        )
    return prompt


def _encode_image(image_input):
    """Encode image from file path (str) or PIL Image to base64"""
    if isinstance(image_input, str):
        return encode_image(image_input)
    else:
        try:
            if image_input.mode != 'RGB':
                image_input = image_input.convert('RGB')
            
            buffered = BytesIO()
            image_input.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding PIL Image: {e}")
            raise


def run_model_inference(prompt, image, model=MODEL_NAME):
    messages = []
    content = [{"type": "text", "text": prompt}]
    
    if image is not None:
        try:
            img_b64 = _encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        except Exception as e:
            print(f"Error encoding image: {e}")
            return "ERROR_IMG"

    messages.append({"role": "user", "content": content})

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR_API"


def process_dataset(ds, lang="en"):
    dataset_split = ds[lang]
    output_file = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{lang}.json")
    
    print(f"\nProcessing '{lang}' split with {len(dataset_split)} samples...")
    results = []
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
                print(f"Resumed from {len(results)} samples.")
            except json.JSONDecodeError:
                print("Warning: Could not read existing results, starting fresh.")
                results = []
    
    start_idx = len(results)
    
    for i in tqdm(range(start_idx, len(dataset_split)), 
                  desc=f"Inference ({lang})", 
                  initial=start_idx, 
                  total=len(dataset_split)):
        data = dataset_split[i]
        
        image = data['image']  # PIL Image
        question = data['question']
        choices = data['choices']
        
        prompt = make_vqa_prompt(question, choices, lang=lang)
        
        response = "ERROR"
        for attempt in range(3):
            response = run_model_inference(prompt, image)
            if response not in ["ERROR_API", "ERROR_IMG"]:
                break
            print(f"Retrying sample {i} (attempt {attempt + 1})...")
            time.sleep(2)
            
        results.append(response)

        if len(results) % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)
    print(f"Finished '{lang}' split. Results saved to {output_file}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading dataset from Hugging Face...")
    try:
        ds = load_dataset("UniParser/RxnBench")
        print(f"Dataset loaded successfully:")
        print(f"  - EN: {len(ds['en'])} samples")
        print(f"  - ZH: {len(ds['zh'])} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    process_dataset(ds, "en")
    process_dataset(ds, "zh")


if __name__ == "__main__":
    main()
