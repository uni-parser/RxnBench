from dotenv import load_dotenv
load_dotenv()

import os

from rxnbench_doc_eval.example_inference import main as main_doc_eval
from rxnbench_eval.example_inference import main as main_img_eval
from rxnbench_doc_eval.evaluate import main as main_doc_eval_evaluate
from rxnbench_eval.evaluate import main as main_img_eval_evaluate


if __name__ == "__main__":
    main_doc_eval()
    main_doc_eval_evaluate(language="en")
    main_doc_eval_evaluate(language="zh")
    main_img_eval()
    main_img_eval_evaluate(language="en")
    main_img_eval_evaluate(language="zh")
