# Kharchify AI/ML Intern Task: Write-up

## Part A: Dish category classifier
**Approach.** Cleaned the data (trim and collapse spaces, title-case labels, case-insensitive de-duplication, conflicting labels dropped, missing rows dropped; the supplied file was already clean, so nothing was removed). Stratified 80/20 split (104 train / 26 test, seed 42). Primary model: **word + character n-gram TF-IDF → Logistic Regression**; `C` was tuned by CV on the *train split only*, and the test set was used once. Character n-grams help with sub-words and spelling variants ("Roomali"/"Rumali"). I compared two other models (word-only LR, LinearSVC).

**Results.**
| Model | Test acc. (26 dishes) | 5-fold CV ×3 (130 dishes) |
|---|---|---|
| LR + word TF-IDF | 0.577 | 0.731 ± 0.066 |
| **LR + word+char TF-IDF (primary)** | **0.577** | **0.762 ± 0.063** |
| LinearSVC + word+char | 0.577 | 0.762 ± 0.060 |

The test score is **57.7 % (15/26)**. I don't think it is the best estimate: one error moves it 3.8 points, and repeated CV gives **≈ 76 %**. Bread and Rice are perfect on test (names contain "naan/roti/rice/biryani"); Beverage recall was 0/4. Char n-grams add ~3 points over word-only; LR and SVC tie, and other n-gram/NB settings tried via CV topped out at ≈ 0.76, so the limit is the data, not the model.

**Where and why it errs.** (1) *No lexical clue*: Samosa, Kheer, Falooda, Kulfi need food knowledge 130 names can't teach. (2) *Tokens shared across classes*: "Masala" appears in Main Course dishes, so Masala Chai/Masala Soda → Main Course; "Tikka" is Starter, but Chicken Tikka *Masala* is a Main; "Chocolate" pulls Chocolate Shake to Dessert. (3) *Genuinely fuzzy labels*: Fruit Salad, Falooda, Tandoori Chicken (starter or main?). (4) Roomali/Rumali Roti are near-duplicates, so a split can leak and flatter the score slightly.

`predict_category()` returns the category, confidence and top-3, and flags `needs_review` below 0.50 so the UI asks the owner instead of silently auto-filling (e.g. "Rava Dosa", which has no matching category, gets 0.30 and is flagged).

## Part B: Sales Q&A
**Approach.** Rule-based, in two stages: `parse_question` (regex/keywords → a structured `Query`: intent, date range, dishes, category) then `execute` (pandas). **All arithmetic is pandas; no language model is involved**, so numbers cannot be invented and there are no API keys. Supports top/bottom-N, revenue, units per dish/category/keyword (e.g. "naan"), best/worst day, and common periods or dates. Fuzzy dish matching handles typos ("panner butter masla") but is strict (same word count, every word close, numbers exact) because the first version matched "chicken wings" to "Chicken 65". The unit tests found that. Unsupported or out-of-range questions get a polite fallback that says why (no customers/orders/costs in the data, no forecasts, unknown dish word). Answers are checked in `tests/` against an independent `csv` calculation (10 pass).

**Assumptions (stated in output).** `price` is a unit price (constant per item), so revenue = quantity × price. Data ends 2026-11-17, so "today" = latest date in the file (override with `--today`). "This week" = Monday → today, so right now it covers only 2 days; "last 7 days" is the rolling alternative.

## What I'd do with more time or data
- **A:** Several hundred to thousands of real menu items, with ingredient/description fields. Add pretrained multilingual sentence embeddings (which know "Kheer" is sweet), a confidence threshold with human-in-the-loop correction, and group-aware splits to stop near-duplicate leakage.
- **B:** An optional LLM that only emits the same JSON `Query` (validated against a schema, code still computes), to cover free-form phrasing; multi-turn follow-ups; a larger labelled question set; order-level data so "orders"/"customers" become answerable.
