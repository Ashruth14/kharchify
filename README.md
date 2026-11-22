# Kharchify: AI/ML Intern Task (Python 3.10+)

```
part_a_classifier.py   Part A: clean -> split -> TF-IDF + LogReg (+2 comparison models) -> report -> predict_category()
part_b_sales_qa.py     Part B: rule-based parser + pandas executor; CLI loop, --demo, --chart, -q
tests/test_part_b.py   Part B answers checked against an independent csv-module calculation
data/                  dishes.csv, sales.csv (as provided)
outputs/               part_a_results.txt, confusion_matrix.png, part_b_examples.txt, sales_charts.png, dish_classifier.joblib
WRITEUP.md             the 1-page write-up
```

## Run
```bash
pip install -r requirements.txt
python part_a_classifier.py                  # Part A report (seed 42, reproducible)
python part_b_sales_qa.py                    # interactive: ask questions, blank line to quit
python part_b_sales_qa.py --demo --chart     # example Q&A + charts
python part_b_sales_qa.py -q "top 3 dishes yesterday" --today 2026-11-17
python -m pytest -q tests
```
```python
from part_a_classifier import predict_category
predict_category("Prawn Biryani")   # {'category': 'Rice', 'confidence': 0.98, 'needs_review': False, ...}
from part_b_sales_qa import SalesQA
SalesQA().ask("How many butter naan were sold last week?")
```
No API keys or network access are needed (no LLM is used).
