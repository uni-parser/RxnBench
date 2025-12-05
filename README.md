# RxnBench: Benchmark for Chemical Reaction Figure Understanding

[`UniParser/RxnBench`](https://huggingface.co/datasets/UniParser/RxnBench)
[`UniParser/RxnBench-Doc`](https://huggingface.co/datasets/UniParser/RxnBench-Doc)

## 📘 Benchmark Summary

RxnBench is a visual question answering (VQA) benchmark comprising 1,525 multiple-choice questions (MCQs) at the PhD-level of organic chemistry reaction understanding. 

The benchmark is built from 305 scientific figures drawn from high-impact OpenAssess journals. 
For each figure, domain experts carefully designed five multiple-choice VQA questions targeting the interpretation of organic reaction diagrams.
These questions were further refined through multiple rounds of rigorous review and revision to ensure both clarity and scientific accuracy. 
The questions cover a variety of types, including the description of chemical reaction images, extraction of reaction content, recognition of molecules or Markush structures, and determination of mechanisms.
This benchmark challenges visual-language models on their foundational knowledge of organic chemistry, multimodal contextual reasoning, and chemical reasoning skills. 


The benchmark is released in both English and Chinese versions. 

## How to run

### Prerequisites

1. **Python Environment**: Python 3.8+ with required dependencies
2. **API Keys**: OpenAI(Or other compatible services) API key for model inference and evaluation
3. **Data Setup**: Ensure data files are properly placed

### Installation

```bash
# Install
pip install -e .
```

### Environment Setup

```bash
export MODEL_NAME="your-model-name"           # e.g., "gpt-4o", "Qwen3-VL-2B-Instruct"
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="your-base-url"         # optional, defaults to OpenAI
export INFER_OUTPUT_DIR="./results"            # output directory
export BASE_PATH="/path/to/rxnbench/data"      # path to RxnBench data directory containing "pdf_files" and "images"(see below)
```

### Running Evaluations


#### Benchmark 1: Regular Evaluation (Image-based VQA)

- **VQA Evaluation**: [`UniParser/RxnBench`](https://huggingface.co/datasets/UniParser/RxnBench)

```bash
# Run inference for English and Chinese
cd rxnbench_eval
python example_inference.py

# Run evaluation
python evaluate.py --models "gpt-4o" --langs "en,zh"
```

**Note**: Due to legal considerations, the actual PDF files for the document evaluation are not provided in our dataset and must be collected and prepared by the user.

#### PDF File Preparation for Document Evaluation

To run the document evaluation benchmark, you need to prepare the corresponding PDF files for each paper referenced in the dataset. Here's how to do it:

1. **Identify Required PDFs**: The dataset contains a `pdf_doi` field for each question, which contains the DOI (Digital Object Identifier) of the paper.

2. **Download PDFs**: You can download the PDFs using the DOI from academic databases or publishers. Common sources include:
   - Publisher websites (ACS, RSC, Wiley, etc.)
   - Academic databases (PubMed, Google Scholar, etc.)
   - Institutional access through universities/libraries

3. **File Organization**: Create a directory structure as follows:
   ```
   BASE_PATH/
   ├── pdf_files/
   │   ├── 10.1021_jacsau.3c00814.pdf
   │   ├── 10.1021_ja123456.pdf
   │   └── ...(naming with data["pdf_doi"].replace("/", "_") as basename)
   └── images/
       ├── question images unzipped from https://huggingface.co/datasets/UniParser/RxnBench-Doc/resolve/main/images.zip
   ```

**Important Notes**:
- Ensure you have proper access rights to download and use the PDFs

#### Benchmark 2: Document Evaluation (Document VQA)

- **Document QA Evaluation**: [`UniParser/RxnBench-Doc`](https://huggingface.co/datasets/UniParser/RxnBench-Doc)

```bash
# Run inference
cd rxnbench_doc_eval
python example_inference.py

# Run evaluation
python evaluate.py
```

### Output Files

- `{MODEL_NAME}_{lang}.json`: Raw model predictions
- `{MODEL_NAME}_{lang}_extracted.jsonl`: Processed predictions with accuracy
- `{MODEL_NAME}_{lang}_accuracy.json`: Accuracy statistics by question type
- `{MODEL_NAME}_{lang}_error.jsonl`: Failed predictions and errors

## 📑 Task Types

We categorize chemical reaction visual question answering tasks into six types:

- **Type 0 — Fact Extraction**: Direct retrieval of textual or numerical information from reaction schemes.
- **Type 1 — Reagent Roles and Functions Identification**: Identification of reagents and their functional roles, requiring chemical knowledge and reaction-type awareness.
- **Type 2 — Reaction Mechanism and Process Understanding**: Interpretation of reaction progression, including intermediates, catalytic cycles, and mechanistic steps.
- **Type 3 — Comparative Analysis and Reasoning**: Comparative evaluation, causal explanation, or outcome prediction under varying conditions.
- **Type 4 — Multi-step Synthesis and Global Understanding**: Comprehension of multi-step pathways, step-to-step coherence, and overall synthetic design.
- **Type 5 — Chemical Structure Recognition**: Extraction and reasoning-based parsing of chemical structures in SMILES or E-SMILES (as defined in the [MolParser](https://arxiv.org/abs/2411.11098) paper).

![output3](https://cdn-uploads.huggingface.co/production/uploads/65f7f16fb6941db5c2e7c4bf/oTOMcZE7oz-Pv4fUUpi0J.png)


## 🎯 Benchmark Evaluation

This benchmark evaluates model performance on multiple-choice question answering (MCQ) tasks.

We provide two versions of the prompt template, depending on the language setting.

**English Prompt**

```
Question: {question}
Choices:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Based on the image and the question, choose the most appropriate answer.
**Only output a single letter (A, B, C, or D)**. Do NOT output any other text or explanation.
```

**Chinese Prompt**

```
问题: {question}
选项:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

请根据图像和问题，从以上四个选项中选择最合适的答案。
只输出单个字母 (A, B, C 或 D)，不要输出选项内容，也不要输出任何解释。
```

**Evaluation Protocol**

If the model’s output is not one of A, B, C, or D, we use GPT-4o to map the output to A–D based on the option content. 
The final evaluation reports the absolute accuracy of the benchmark in both English and Chinese versions.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct.

## 📄 License

See the LICENSE file for details.

## 📖 Citation

Our paper is coming soon. Please cite this repository for now:

```bibtex
@misc{rxnbench2025,
  title={RxnBench: A Benchmark for Chemical Reaction Figure Understanding},
  author={UniParser Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/uni-parser/RxnBench}
}
```