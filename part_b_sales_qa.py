"""Part B - Plain-English questions over sales.csv (Kharchify AI/ML Intern task).

Usage
  python part_b_sales_qa.py                 # interactive CLI loop
  python part_b_sales_qa.py --demo          # run the example questions, save outputs/part_b_examples.txt
  python part_b_sales_qa.py --chart         # save outputs/sales_charts.png
  python part_b_sales_qa.py -q "top 3 dishes yesterday" [--today 2026-11-17]

Architecture (deliberately two-stage, so an LLM could replace stage 1 later)
  1. parse_question(text) -> Query    pure text -> structured query (intent, dates, dish, ...)
  2. execute(query)       -> answer   ALL arithmetic is done by pandas, never by a language model.
Rule-based parsing is used: it is deterministic, free, offline, needs no API key, and cannot
invent numbers. Anything it cannot map to a supported query gets a polite fallback.

"Today": the CSV ends on 2026-11-17, so "today" defaults to the LATEST DATE IN THE DATA (not the
wall clock - otherwise every answer would be empty). Override with --today YYYY-MM-DD.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
SALES_CSV = BASE / "data" / "sales.csv"
DISHES_CSV = BASE / "data" / "dishes.csv"
OUT = BASE / "outputs"
CUR = "₹"
FUZZY_CUTOFF = 0.80

HINT = ("I can answer questions about sales, e.g. \"top 3 dishes yesterday\", \"how much did we sell "
        "today?\", \"how many butter naan were sold this week?\", \"revenue from desserts last week\" "
        "or \"which day had the highest sales?\".")
FALLBACK = "Sorry, I couldn't understand that one, and I won't guess. " + HINT

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
WEEKDAYS = {d: i for i, d in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
CATEGORY_WORDS = {"starter": "Starter", "starters": "Starter", "appetizer": "Starter",
                  "appetizers": "Starter", "main course": "Main Course", "main courses": "Main Course",
                  "mains": "Main Course", "dessert": "Dessert", "desserts": "Dessert",
                  "bread": "Bread", "breads": "Bread", "rice": "Rice", "beverage": "Beverage",
                  "beverages": "Beverage", "drink": "Beverage", "drinks": "Beverage"}
STOP = set("""a an the of in on at for to is are was were be been we our us i my me did do does
how many much what which who show tell give list sold sell selling sale sales total number count
quantity qty units unit portions plates orders items dishes dish food week day today yesterday
this last past month year by from with and or per revenue income earned earn made make top best
bottom worst most least highest lowest popular please can you get had have has that it its
overall altogether all so far there any some just about around roughly category categories seller
sellers day days daily average avg mean per last past previous ago cumulative combined during within across only also then now currently yet ever""".split())


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_sales(path: Path = SALES_CSV, dishes_path: Path = DISHES_CSV) -> pd.DataFrame:
    """Load + clean sales; add revenue (= quantity x unit price) and category."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["item"] = df["item"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "item", "quantity", "price"])   # drop unusable rows
    df = df[(df["quantity"] > 0) & (df["price"] >= 0)].copy()
    df["quantity"] = df["quantity"].astype(int)
    df["revenue"] = df["quantity"] * df["price"]       # price is a per-unit price (constant per item)
    cats = pd.read_csv(dishes_path)
    cats["key"] = cats["dish_name"].str.strip().str.lower()
    cat_map = cats.drop_duplicates("key").set_index("key")["category"].str.strip().str.title()
    df["category"] = df["item"].str.lower().map(cat_map).fillna("Unknown")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Structured query
# --------------------------------------------------------------------------- #
@dataclass
class Query:
    intent: str                       # rank_items | rank_categories | rank_days | revenue | quantity | fallback
    start: date | None = None
    end: date | None = None
    period_label: str = "all recorded data"
    dishes: list[str] = field(default_factory=list)   # resolved item names
    dish_note: str = ""                                # how the dishes were matched
    category: str | None = None
    n: int = 5
    metric: str = "quantity"          # quantity | revenue (ranking metric)
    ascending: bool = False           # True for "least/worst"
    note: str = ""                    # reason text for fallback


# --------------------------------------------------------------------------- #
# Stage 1: parsing
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[?!,;]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _fmt(d: date) -> str:
    return f"{d:%a} {d.isoformat()}"


def parse_period(t: str, today: date) -> tuple[date | None, date | None, str, bool]:
    """Return (start, end, label, found). found=False -> no time expression (use all data)."""
    one = lambda d, lab: (d, d, f"on {_fmt(d)}" if lab == "on" else f"{lab} ({_fmt(d)})", True)
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return one(date(int(m[1]), int(m[2]), int(m[3])), "on")
        except ValueError:
            return None, None, "invalid date", True
    try:
        for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)? (?:of )?([a-z]{3})[a-z]*(?: (\d{4}))?\b", t):
            if m[2] in MONTHS:                                   # "5 nov", "5th of november 2026"
                return one(date(int(m[3] or today.year), MONTHS[m[2]], int(m[1])), "on")
        for m in re.finditer(r"\b([a-z]{3})[a-z]* (\d{1,2})(?:st|nd|rd|th)?(?: (\d{4}))?\b", t):
            if m[1] in MONTHS:                                   # "nov 5", "november 5th"
                return one(date(int(m[3] or today.year), MONTHS[m[1]], int(m[2])), "on")
    except ValueError:
        return None, None, "invalid date", True

    if "day before yesterday" in t:
        return one(today - timedelta(days=2), "day before yesterday")
    if "yesterday" in t:
        return one(today - timedelta(days=1), "yesterday")
    if re.search(r"\btoday|tonight|so far today\b", t):
        return one(today, "today")
    m = re.search(r"\b(?:last|past|previous) (\d+|" + "|".join(NUMBER_WORDS) + r") days\b", t)
    if m:
        n = int(m[1]) if m[1].isdigit() else NUMBER_WORDS[m[1]]
        s = today - timedelta(days=n - 1)
        return s, today, f"last {n} days ({s.isoformat()} to {today.isoformat()})", True
    if re.search(r"\b(?:past|last) 7 days|past week\b", t):
        s = today - timedelta(days=6)
        return s, today, f"last 7 days ({s.isoformat()} to {today.isoformat()})", True
    if re.search(r"\blast week\b", t):
        s = today - timedelta(days=today.weekday() + 7)
        e = s + timedelta(days=6)
        return s, e, f"last week (Mon {s.isoformat()} to Sun {e.isoformat()})", True
    if re.search(r"\bthis week\b", t):
        s = today - timedelta(days=today.weekday())
        return s, today, f"this week (Mon {s.isoformat()} to {_fmt(today)})", True
    if re.search(r"\blast month\b", t):
        e = today.replace(day=1) - timedelta(days=1)
        return e.replace(day=1), e, f"last month ({e:%B %Y})", True
    if re.search(r"\bthis month\b", t):
        return today.replace(day=1), today, f"this month ({today:%B %Y} so far)", True
    m = re.search(r"\b(?:on |last |this )?(" + "|".join(WEEKDAYS) + r")\b", t)
    if m:   # most recent such weekday on or before "today"
        d = today - timedelta(days=(today.weekday() - WEEKDAYS[m[1]]) % 7)
        return one(d, f"on {m[1].capitalize()}")
    return None, None, "all recorded data", False


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def resolve_dishes(t: str, items: list[str]) -> tuple[list[str], str, list[str]]:
    """Map the question to menu items: exact name > fuzzy name > shared keyword (e.g. 'naan')."""
    lower = {i.lower(): i for i in items}
    padded = f" {t} "
    # 1) exact: longest dish name contained in the question (handles plural 's')
    hits = [n for n in lower if re.search(rf"(?<![a-z]){re.escape(n)}(?:e?s)?(?![a-z])", padded)]
    if hits:
        longest = max(hits, key=len)
        # drop names that are substrings of the longest ("kulcha" inside "paneer kulcha")
        hits = [h for h in hits if h == longest or h not in longest]
        return [lower[h] for h in hits], "exact name", []
    # 2) fuzzy: slide 1..5-word windows over the question; a dish matches only if it has the same
    #    number of words, EVERY word is close to its counterpart, and numbers match exactly
    #    (so "chicken wings" can never be mistaken for "Chicken 65").
    toks = _tokens(t)
    best, best_score = None, 0.0
    for size in range(1, 6):
        for i in range(len(toks) - size + 1):
            wtoks = toks[i:i + size]
            window = " ".join(wtoks)
            if size == 1 and (len(window) < 5 or window in STOP or window in CATEGORY_WORDS):
                continue
            for name in lower:
                ntoks = name.split()
                if len(ntoks) != size:
                    continue
                pairs = list(zip(wtoks, ntoks))
                if any((a.isdigit() or b.isdigit()) and a != b for a, b in pairs):
                    continue
                if min(difflib.SequenceMatcher(None, a, b).ratio() for a, b in pairs) < 0.7:
                    continue
                sc = difflib.SequenceMatcher(None, window, name).ratio()
                if sc > best_score:
                    best, best_score = name, sc
    if best and best_score >= FUZZY_CUTOFF:
        return [lower[best]], f"interpreted your dish name as '{lower[best]}'", []
    # 3) generic keyword shared by several dishes, e.g. "naan", "biryani", "paneer"
    words = [w for w in toks if w not in STOP and w not in CATEGORY_WORDS and len(w) > 2
             and not w.isdigit() and w not in MONTHS and w not in WEEKDAYS
             and not re.fullmatch(r"\d+(st|nd|rd|th)", w)]
    matched, group = [], []
    for w in words:
        g = [orig for low, orig in lower.items() if _singular(w) in [_singular(x) for x in _tokens(low)]]
        if g:
            matched.append(w)
            group += [x for x in g if x not in group]
    unknown = [w for w in words if w not in matched]
    if group and not unknown:
        return group, f"matched all items containing {' & '.join(repr(w) for w in matched)}", []
    return [], "", unknown


def parse_question(text: str, items: list[str], today: date) -> Query:
    t = _clean(text)
    if not t or len(t) < 3:
        return Query("fallback", note="empty question")
    start, end, label, found = parse_period(t, today)
    if label == "invalid date":
        return Query("fallback", note="that date doesn't look valid")
    q = Query("fallback", start=start, end=end, period_label=label)

    # number requested ("top 3", "five best")
    m = re.search(r"\b(?:top|bottom|best|worst|first|last)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\b", t) \
        or re.search(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:best|top|worst|least|most|popular)\b", t)
    n_given = None
    if m:
        n_given = int(m[1]) if m[1].isdigit() else NUMBER_WORDS[m[1]]
    # NOTE: if the user gave a number, n_given stays; otherwise defaults are set per intent below

    ranking = re.search(r"\b(top|best|bottom|worst|least|most|highest|lowest|popular|bestsell\w*|best-sell\w*|"
                        r"favou?rite|busiest|slowest|biggest)\b", t)
    asc = bool(re.search(r"\b(bottom|worst|least|lowest|slowest|poorest)\b", t))
    metric = "revenue" if re.search(r"\b(revenue|earning\w*|income|money|rupees|sales value|by sales)\b", t) \
        else "quantity"
    singular_dish = bool(re.search(r"\bdish\b|\bitem\b", t)) and not re.search(r"\bdishes\b|\bitems\b", t)

    # --- things this dataset cannot answer ---
    if re.search(r"\btomorrow\b|\bnext (week|month|day)\b|\bforecast\b|\bpredict\w*\b|\bexpect\w*\b", t):
        q.note = "I can only report past sales from the file, not forecasts"
        return q
    if re.search(r"\border\w*\b|\bcustomers?\b|\bbills?\b|\btables?\b|\bprofit\b|\bmargin\b|\bstock\b|"
                 r"\binventory\b|\bwaste\b|\bstaff\b|\bcost\b|\bexpens\w+\b", t):
        q.note = ("the data only has item, quantity and price per day, so it has no order counts, "
                  "customers, costs or stock levels")
        return q

    # --- which day (best/worst day) ---
    if re.search(r"\b(which|what) (day|date)\b|\b(busiest|best|worst|slowest) day\b|\bday (with|had)\b", t) \
            and re.search(r"\b(sales|sold|sell\w*|revenue|busiest|highest|lowest|best|worst|most|least|slowest)\b", t):
        q.intent, q.ascending, q.n = "rank_days", asc, n_given or 1
        q.metric = "quantity" if re.search(r"\b(quantity|units|items sold|sold the most)\b", t) else "revenue"
        return q

    cat = next((v for k, v in sorted(CATEGORY_WORDS.items(), key=lambda kv: -len(kv[0]))
                if re.search(rf"\b{re.escape(k)}\b", t)), None)
    dishes, how, unknown = resolve_dishes(t, items)
    is_group = how.startswith("matched all items")
    # a bare category word ("desserts", "breads", "rice") beats a keyword-group guess
    if dishes and is_group and cat:
        dishes, how, is_group = [], "", False
    if not dishes and unknown and not cat and not ranking:
        q.note = f"I couldn't match '{unknown[0]}' to any dish or category"
        return q

    strong_qty = bool(re.search(r"\bhow many\b|\bunits\b|\bquantity\b|\bportions\b|\bplates\b|"
                                r"\bnumber of\b|\bcount\b", t))
    strong_money = bool(re.search(r"\bhow much\b|\brevenue\b|\bsales\b|\bearn\w*\b|\bincome\b|\bmade\b|"
                                  r"\bturnover\b|\bcollection\b|\bbusiness\b|\bsold for\b", t))
    soft = bool(re.search(r"\bsold\b|\bsell\w*\b", t))

    # --- ranking questions (single exact dish + ranking word is treated as a plain lookup) ---
    if ranking and (not dishes or is_group):
        if re.search(r"\bcategor(y|ies)\b", t):
            q.intent, q.metric, q.n, q.ascending = "rank_categories", metric, n_given or 1, asc
            return q
        q.intent, q.metric, q.ascending = "rank_items", metric, asc
        q.n = n_given or (1 if singular_dish else 5)
        q.category, q.dishes, q.dish_note = (None if dishes else cat), dishes, how
        return q

    # --- quantity / revenue for a dish, a category, or everything ---
    if dishes or cat:
        q.dishes, q.dish_note, q.category = dishes, how, (None if dishes else cat)
        lead_qty = strong_qty and not strong_money
        q.intent = "quantity" if lead_qty else "revenue"
        q.metric = q.intent
        return q
    if strong_qty and re.search(r"\b(items|dishes|units|portions|plates)\b", t):
        q.intent, q.metric = "quantity", "quantity"
        return q
    if strong_money or (soft and not strong_qty) or re.search(r"\b(average|avg)\b", t):
        q.intent, q.metric = "revenue", "revenue"
        return q
    q.note = "I couldn't map it to a sales question"
    return q


# --------------------------------------------------------------------------- #
# Stage 2: execution (pure pandas)
# --------------------------------------------------------------------------- #
def _money(x: float) -> str:
    return f"{CUR}{x:,.0f}"


def _plural(n: int, w: str) -> str:
    return f"{n} {w}" + ("" if n == 1 else "s")


def execute(q: Query, df: pd.DataFrame, today: date, raw_text: str = "") -> str:
    if q.intent == "fallback":
        return FALLBACK if not q.note else f"Sorry, I can't answer that: {q.note}. I won't guess. " + HINT
    lo, hi = df["date"].min(), df["date"].max()
    sub = df
    if q.start is not None:
        sub = df[(df["date"] >= q.start) & (df["date"] <= q.end)]
    span = f"(sales data covers {lo.isoformat()} to {hi.isoformat()})"
    if sub.empty:
        tip = " Tip: try 'last 7 days' for a rolling window." if q.period_label.startswith("this week") else ""
        return f"No sales recorded {q.period_label} {span}.{tip}"
    when = q.period_label if q.start is not None else f"across all recorded data ({lo.isoformat()} to {hi.isoformat()})"
    days_in_data = sub["date"].nunique()
    partial = ""
    if q.start is not None and (q.start < lo or q.end > hi):
        partial = f" Note: the data only covers part of that period ({lo.isoformat()} to {hi.isoformat()})."

    if q.intent == "rank_days":
        g = sub.groupby("date")[["quantity", "revenue"]].sum().sort_values(q.metric, ascending=q.ascending)
        g = g.head(q.n)
        word = "Lowest" if q.ascending else "Highest"
        lines = [f"{word} {'sales' if q.metric == 'revenue' else 'volume'} day {when}:"]
        for d, r in g.iterrows():
            lines.append(f"  {_fmt(d)}: {_money(r.revenue)} ({int(r.quantity)} items)")
        return "\n".join(lines) + partial

    if q.intent == "rank_categories":
        g = sub.groupby("category")[["quantity", "revenue"]].sum().sort_values(q.metric, ascending=q.ascending)
        g = g.head(q.n)
        lines = [f"{'Weakest' if q.ascending else 'Top'} categor{'y' if len(g) == 1 else 'ies'} {when} by {q.metric}:"]
        for i, (c, r) in enumerate(g.iterrows(), 1):
            lines.append(f"  {i}. {c}: {int(r.quantity)} sold, {_money(r.revenue)}")
        return "\n".join(lines) + partial

    if q.intent == "rank_items":
        s = sub[sub["category"] == q.category] if q.category else sub
        if q.dishes:
            s = s[s["item"].isin(q.dishes)]
        if s.empty:
            return f"No sales recorded for {q.category or 'those items'} {when} {span}."
        g = s.groupby("item")[["quantity", "revenue"]].sum().reset_index()
        # tie-break by the other metric, then name, so output is deterministic
        other = "revenue" if q.metric == "quantity" else "quantity"
        g = g.sort_values([q.metric, other, "item"], ascending=[q.ascending, q.ascending, True])
        top = g.head(q.n)
        what = f"{q.category} " if q.category else ""
        pos = "Bottom" if q.ascending else "Top"
        head = f"{pos} {len(top)} {what}dishes {when} by {q.metric}:" if len(top) > 1 \
            else f"{pos} {what}dish {when} by {q.metric}:"
        lines = [head] + [f"  {i}. {r.item}: {int(r.quantity)} sold, {_money(r.revenue)}"
                          for i, r in enumerate(top.itertuples(), 1)]
        tied = g.iloc[len(top):][g.iloc[len(top):][q.metric] == top[q.metric].iloc[-1]]
        if len(tied):
            lines.append(f"  (note: {len(tied)} more dish{'es' if len(tied) > 1 else ''} tie on "
                         f"{q.metric} = {top[q.metric].iloc[-1]:,.0f}; shown order uses the other metric)")
        return "\n".join(lines) + partial

    # quantity / revenue
    scope, extra = "all items", ""
    if q.dishes:
        sub = sub[sub["item"].isin(q.dishes)]
        scope = " + ".join(q.dishes) if len(q.dishes) <= 3 else f"{len(q.dishes)} items"
        if q.dish_note.startswith("matched all"):
            scope = "Items matching " + q.dish_note.split("containing ")[1]
            per = sub.groupby("item")["quantity"].sum().sort_values(ascending=False)
            extra = "\n  Breakdown: " + ", ".join(f"{i} {int(n)}" for i, n in per.items()) if len(per) else ""
        elif q.dish_note.startswith("interpreted"):
            extra = f"\n  ({q.dish_note})"
    elif q.category:
        sub = sub[sub["category"] == q.category]
        scope = f"{q.category} category"
    if sub.empty:
        tip = " Tip: try 'last 7 days' for a rolling window." if when.startswith("this week") else ""
        return f"No sales of {scope} recorded {when} {span}.{tip}"
    qty, rev = int(sub["quantity"].sum()), float(sub["revenue"].sum())
    if q.intent == "quantity":
        return f"{scope}: {_plural(qty, 'unit')} sold {when} (worth {_money(rev)}).{extra}{partial}"
    msg = f"Sales for {scope} {when}: {_money(rev)} from {_plural(qty, 'unit')} sold."
    if q.start is None or (q.end - q.start).days >= 1:
        n_days = sub["date"].nunique()
        if n_days > 1:
            msg += f" Average {_money(rev / n_days)} per day with sales ({n_days} days)."
    return msg + extra + partial


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class SalesQA:
    def __init__(self, df: pd.DataFrame | None = None, today: date | str | None = None):
        self.df = load_sales() if df is None else df
        self.items = sorted(self.df["item"].unique())
        latest = self.df["date"].max()
        self.today = latest if today is None else (date.fromisoformat(today) if isinstance(today, str) else today)

    def ask(self, question: str) -> str:
        try:
            q = parse_question(question, self.items, self.today)
            return execute(q, self.df, self.today, question)
        except Exception as exc:  # never crash the assistant; never fabricate
            return f"Sorry, something went wrong while answering that ({type(exc).__name__}). Please rephrase."


EXAMPLES = [
    "What were the top 3 dishes yesterday?",
    "How much did we sell today?",
    "How many butter naan were sold this week?",
    "How many butter naan were sold last week?",
    "How many butter naan were sold in the last 7 days?",
    "Top 5 dishes in the last 7 days",
    "Revenue from desserts last week",
    "Which day had the highest sales?",
    "How many biryani did we sell on 2026-11-10?",
    "how many panner butter masla were sold in the last 10 days",
    "What is the best selling category this month?",
    "Least popular dish last week",
    "What's the weather like tomorrow?",
    "How many customers came today?",
    "How many chicken wings did we sell?",
    "Tell me a joke",
    "Total sales on 2025-01-01",
    "",
]


def make_chart(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    daily = df.groupby("date")["revenue"].sum()
    ax[0].plot(pd.to_datetime(daily.index), daily.values, marker="o", color="#2a6f97")
    ax[0].set_title("Daily sales revenue"); ax[0].set_ylabel(f"Revenue ({CUR})")
    ax[0].tick_params(axis="x", rotation=45); ax[0].grid(alpha=.3)
    cat = df.groupby("category")["revenue"].sum().sort_values()
    ax[1].barh(cat.index, cat.values, color="#e07a5f")
    ax[1].set_title("Total revenue by category"); ax[1].set_xlabel(f"Revenue ({CUR})")
    for i, v in enumerate(cat.values):
        ax[1].text(v, i, f" {CUR}{v:,.0f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Ask plain-English questions about sales.csv")
    ap.add_argument("-q", "--question"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--today", help="override 'today' (YYYY-MM-DD); default = latest date in the data")
    a = ap.parse_args()
    qa = SalesQA(today=a.today)
    OUT.mkdir(exist_ok=True)
    header = f"[Reference date 'today' = {_fmt(qa.today)}; data covers {qa.df['date'].min()} to {qa.df['date'].max()}]"

    if a.chart:
        make_chart(qa.df, OUT / "sales_charts.png"); print("saved", OUT / "sales_charts.png")
    if a.demo:
        lines = [header, ""]
        for ex in EXAMPLES:
            lines += [f"Q: {ex!r}", f"A: {qa.ask(ex)}", ""]
        text = "\n".join(lines); print(text)
        (OUT / "part_b_examples.txt").write_text(text, encoding="utf-8")
    elif a.question:
        print(header); print(qa.ask(a.question))
    elif not a.chart:
        print(header); print("Ask a question (blank line or 'quit' to exit).")
        while True:
            try:
                s = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if s.lower() in {"", "q", "quit", "exit"}:
                break
            print(qa.ask(s))


if __name__ == "__main__":
    main()
