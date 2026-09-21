"""Run Simple Jev alone in its supported environment.

Set PYTHONPATH to the Simple Jev checkout and its hf-server directory.
Uses cached Qwen2.5-1.5B, float32, MPS, seven warmed repeats.
"""

import copy
import json
import statistics
import threading
from time import perf_counter

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ClassifierRequest, build_response
from hf_server import HFBackend, PromptCompiler, unique_prompt_tokens

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
STATE = 'I was charged twice for my subscription. Please refund the duplicate payment.'
CRITERIA = {
    'billing': 'Payments, invoices, and refunds',
    'technical': 'Product errors and troubleshooting',
    'sales': 'Pricing and purchases',
    'other': 'Anything else',
}


def payload(question_count):
    return {
        'model': MODEL,
        'state': STATE,
        'questions': {
            f'route_{i}': {
                'type': 'choice',
                'instructions': f'Which team should handle this message? Request {i + 1}.',
                'criteria': copy.deepcopy(CRITERIA),
            }
            for i in range(question_count)
        },
    }


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to('mps').eval()
compiler = PromptCompiler(tokenizer, max_tokens=min(model.config.max_position_embeddings, tokenizer.model_max_length))
backend = HFBackend(model, max_batch_size=32, max_batch_tokens=32768)

rows = []
for count in (1, 4, 16):
    request = ClassifierRequest.model_validate(payload(count))

    def run():
        start = perf_counter()
        compiled = compiler.compile(request)
        compiled_at = perf_counter()
        scored = backend._score(compiled, threading.Event())
        scored_at = perf_counter()
        response = build_response(
            compiled.plan,
            scored.logits,
            input_tokens=unique_prompt_tokens([b.token_ids for b in compiled.branches]),
            output_tokens=0,
        )
        return {
            'end_to_end': perf_counter() - start,
            'compile': compiled_at - start,
            'score': scored_at - compiled_at,
            'metrics': scored.metrics,
            'response': response,
        }

    run()
    samples = [run() for _ in range(7)]
    last = samples[-1]
    rows.append({
        'questions': count,
        'median_end_to_end_seconds': statistics.median(x['end_to_end'] for x in samples),
        'median_score_seconds': statistics.median(x['score'] for x in samples),
        'median_compile_seconds': statistics.median(x['compile'] for x in samples),
        'input_tokens': last['response']['usage']['input_tokens'],
        'metrics': last['metrics'],
        'answers': [a['choice'] for a in last['response']['answers'].values()],
    })

print(json.dumps({
    'model': MODEL,
    'device': 'mps',
    'dtype': str(next(model.parameters()).dtype),
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'repeats': 7,
    'rows': rows,
}, indent=2))
