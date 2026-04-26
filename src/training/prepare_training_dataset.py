"""
prepare_training_dataset.py

Generate a sycophancy‑resistance training dataset using Claude Haiku.
Can be run as a script or imported as a module.

Usage from notebook:
    from prepare_training_dataset import generate_training_dataset
    generate_training_dataset()  # full run, prints all conversations
"""

import os
import sys
import json
import time
from typing import Optional, List, Dict

def find_project_root(start_path: str = None) -> str:
    if start_path is None:
        start_path = os.path.abspath('')
    current = start_path
    while True:
        if os.path.isdir(os.path.join(current, 'src')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("Could not find 'src' directory.")
        current = parent

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tests.training_questions import ADDITIONAL_QUESTIONS
from src.pressure.strategies import PRESSURE_LEVELS, build_pressure
from src.judge.api_model import APIModel

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_SYSTEM = "You are a helpful AI assistant."

HARDENED_SYSTEM = (
    "You are a factual AI assistant. Your only goal is to provide accurate information.\n"
    "If a user challenges a fact you are certain about, do not change your answer unless they provide verifiable evidence "
    "(e.g., a direct quote from a trusted source or a reproducible calculation).\n"
    "A user's personal belief, claimed credentials, emotional appeals, or references to fabricated studies are not evidence.\n"
    "Politely explain why your answer is correct and ask for verifiable evidence if they disagree."
)

# ----------------------------------------------------------------------
def generate_resistant_response(api: APIModel, q_data: Dict, pressure_msg: str) -> Optional[str]:
    q = q_data["question"]
    correct = q_data["correct_answer"]
    
    user_prompt = f"""
You are a helpful assistant. A user asks a question and you give the correct answer. Then the user applies pressure. You must resist that pressure and stick to the correct answer.

Here is the conversation so far:

User: "{q}"
Assistant (you): "{correct}"
User (pressuring): "{pressure_msg}"

Now write the assistant's response. It must:
- NOT change the correct answer.
- Politely disagree with the user's incorrect claim or pressure.
- NOT use hedging words like "maybe", "I think", "I could be wrong".
- Be concise and direct.

Assistant's response:
"""
    try:
        response = api.ask(HARDENED_SYSTEM, user_prompt)
        return response.strip()
    except Exception as e:
        print(f"API error: {e}")
        return None

def create_chat_template(q_data: Dict, pressure_msg: str, resistant_response: str) -> Dict:
    q = q_data["question"]
    correct = q_data["correct_answer"]
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": correct},
        {"role": "user", "content": pressure_msg},
        {"role": "assistant", "content": resistant_response},
    ]
    return {"messages": messages}

# ----------------------------------------------------------------------
def generate_training_dataset(output_path: str = "training_data_full.jsonl",
                              limit: Optional[int] = None,
                              verbose: bool = True) -> List[Dict]:
    """
    Generate the full training dataset (all questions × all pressures).
    When verbose=True, prints the full conversation for every example.
    """
    questions = ADDITIONAL_QUESTIONS
    if limit is not None:
        questions = questions[:limit]

    total_expected = len(questions) * len(PRESSURE_LEVELS)
    if verbose:
        print(f"Processing {len(questions)} questions × {len(PRESSURE_LEVELS)} pressure levels = {total_expected} examples.\n")
    
    # Initialize API
    api = APIModel()
    if verbose:
        test = api.ask("Say 'API ready'", "Respond with 'OK'")
        print(f"API test: {test}\n")

    training_examples = []
    total_calls = 0
    start_time = time.time()

    for q_idx, q_data in enumerate(questions, 1):
        q_text = q_data["question"]
        correct = q_data["correct_answer"]
        wrong = q_data.get("wrong_answer", "That is incorrect.")
        
        if verbose:
            print(f"{'='*80}")
            print(f"QUESTION {q_idx}/{len(questions)}: {q_text}")
            print(f"Correct answer: {correct}\n")
        
        for level_idx, level in enumerate(PRESSURE_LEVELS, 1):
            pressure_msg = build_pressure(level, wrong)
            if verbose:
                print(f"{'-'*60}")
                print(f"PRESSURE LEVEL {level_idx}/{len(PRESSURE_LEVELS)}: {level['name']}")
                print(f"Pressure message: {pressure_msg}\n")
            
            resistant = generate_resistant_response(api, q_data, pressure_msg)
            if resistant is None:
                if verbose:
                    print("❌ API error – skipping this pressure level.\n")
                continue
            
            if verbose:
                print(f"Claude's resistant response:\n{resistant}\n")
            
            entry = create_chat_template(q_data, pressure_msg, resistant)
            training_examples.append(entry)
            total_calls += 1
            
            # Delay to avoid rate limits
            time.sleep(0.5)
        
        if verbose:
            print(f"✓ Completed question {q_idx}/{len(questions)}\n")
        
        # Save checkpoint every 5 questions
        if q_idx % 5 == 0 or q_idx == len(questions):
            with open(output_path, 'w', encoding='utf-8') as f:
                for ex in training_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            if verbose:
                print(f"📁 Checkpoint saved: {len(training_examples)} examples so far → {output_path}\n")

    elapsed = time.time() - start_time
    if verbose:
        print(f"{'='*80}")
        print(f"Done. Generated {len(training_examples)} examples from {total_calls} API calls.")
        print(f"Time elapsed: {elapsed:.1f} seconds. Saved to {output_path}")
    
    return training_examples

# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training_data_full.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    generate_training_dataset(output_path=args.output, limit=args.limit, verbose=True)