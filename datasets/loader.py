import json
from typing import List, Dict


def load_multiarith(path: str, limit: int = 50) -> List[Dict]:
    """Load MultiArith dataset and normalize into {question, answer} format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data[:limit]:
        samples.append({
            "question": item["sQuestion"].strip(),
            "answer": str(item["lSolutions"][0])  # take first solution
        })
    return samples


def load_gsm8k(path: str, limit: int = 50) -> List[Dict]:
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data[:limit]
