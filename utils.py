import random
import re
import time
from fractions import Fraction

import numpy as np

from config import SYSTEM_PROMPT


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def bf16_ok():
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


# ---------------------------------------------------------------------------
# Answer parsing (GSM8K numeric answers)
# ---------------------------------------------------------------------------
def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def to_number(s):
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\(mathrm|textbf|mathbf)\{([^}]*)\}", r"\2", s)
    for tok in ["\\$", "$", ",", "\\%", "%", "\\!", "\\,", "\\ ", " ", "dollars", "\\left", "\\right"]:
        s = s.replace(tok, "")
    m = re.fullmatch(r"-?\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s.replace("\\dfrac", "\\frac"))
    try:
        if m:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
        if re.fullmatch(r"-?\d+/\d+", s):
            return float(Fraction(s))
        return float(s.rstrip("."))
    except (ValueError, ZeroDivisionError):
        nums = re.findall(r"-?\d+\.?\d*", s)
        return float(nums[0]) if len(nums) == 1 else None


def extract_pred(text):
    return to_number(last_boxed(text))


def is_correct(text, gold):
    pred = extract_pred(text)
    return pred is not None and abs(pred - gold) < 1e-4 * max(1.0, abs(gold))


def gsm8k_gold(answer_field):
    return float(answer_field.split("####")[-1].strip().replace(",", ""))


def split_steps(solution):
    return [s.strip() for s in solution.split("\n\n") if s.strip()]


def load_gsm8k(n_train, n_test, seed):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    tr = ds["train"].shuffle(seed=seed).select(range(n_train))
    te = ds["test"].shuffle(seed=seed).select(range(n_test))
    fmt = lambda x: dict(question=x["question"], gold=gsm8k_gold(x["answer"]))
    return [fmt(x) for x in tr], [fmt(x) for x in te]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
class PromptBuilder:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def head(self, question):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def with_steps(self, question, steps, open_next=True):
        body = "\n\n".join(steps)
        if steps and open_next:
            body += "\n\n"
        return self.head(question) + body
