# RxnBench: Benchmark for Chemical Reaction Figure/Document Understanding

[🤗`UniParser/RxnBench`](https://huggingface.co/datasets/UniParser/RxnBench) | 
[🤗`UniParser/RxnBench-Doc`](https://huggingface.co/datasets/UniParser/RxnBench-Doc) |
[📚`arxiv`](https://arxiv.org/abs/2512.23565)

## Benchmark Summary

RxnBench is a PhD-level benchmark suite for organic-chemistry Image/PDF VQA, split into two parts:

[RxnBench(SF-QA)](https://huggingface.co/datasets/UniParser/RxnBench): A benchmark for Chemical Reaction Figure Understanding, including 1,525 English/Chinese MCQs built on 305 peer-reviewed chemical reaction figures.

[RxnBench(FD-QA)](https://huggingface.co/datasets/UniParser/RxnBench-Doc): A benchmark for Multimodal Understanding of Chemistry Reaction Literature, including 540 English/Chinese multiple-select questions on document-level chemical reaction understanding.

The benchmark is released in both English and Chinese versions.

This repo provide a sample code to evaluate on this dataset.

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

#### Benchmark 1: RxnBench (SF-QA)

**Single Figure VQA Evaluation:** [`UniParser/RxnBench`](https://huggingface.co/datasets/UniParser/RxnBench)

```bash
# Run inference for English and Chinese
cd rxnbench_eval
python example_inference.py

# Run evaluation
python evaluate.py
```

#### Benchmark 2: RxnBench (FD-QA)

**Full Document VQA Evaluation:** [`UniParser/RxnBench-Doc`](https://huggingface.co/datasets/UniParser/RxnBench-Doc)

**Step1: PDF file preparation**

**Note**: Due to legal considerations, the actual PDF files for the document evaluation are not provided in our dataset and must be collected and prepared by the user.

To run the document evaluation benchmark, you need to prepare the corresponding PDF files for each paper referenced in the dataset:

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

**Step2: Run evaluation**

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

## 📄 License

See the LICENSE file for details.

## 📖 Citation

```bibtex
@article{li2025rxnbench,
  title={RxnBench: A Multimodal Benchmark for Evaluating Large Language Models on Chemical Reaction Understanding from Scientific Literature},
  author={Li, Hanzheng and Fang, Xi and Li, Yixuan and Huang, Chaozheng and Wang, Junjie and Wang, Xi and Bai, Hongzhe and Hao, Bojun and Lin, Shenyu and Liang, Huiqi and Zhang, Linfeng and Ke, Guolin},
  journal={arXiv preprint arXiv:2512.23565},
  year={2025}
}
```
