"""Compare local request processing, candidate scaling, and a synthetic accuracy smoke set.

Excludes HTTP transport, server queues, model loading, and warmup.
"""

import argparse
import copy
import json
import statistics
import sys
import threading
from pathlib import Path
from time import perf_counter

parser = argparse.ArgumentParser(description='Local HF comparison; both implementations use the same model instance. Run in Open Cricket HF environment; Simple Jev runtime compatibility must be checked separately.')
parser.add_argument('--simple-jev', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.repeats < 1:
    parser.error('--repeats must be positive')
ROOT = Path(__file__).resolve().parents[1]
SIMPLE = args.simple_jev.resolve()
sys.path.insert(0, str(SIMPLE))
sys.path.insert(0, str(SIMPLE / 'hf-server'))
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ClassifierRequest, build_response
from hf_server import HFBackend as SimpleBackend
from hf_server import PromptCompiler, unique_prompt_tokens
from open_cricket.hf import HuggingFaceBackend
from open_cricket.sdk import LocalClient

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


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=False).to('mps').eval()

open_backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
open_backend.torch = torch
open_backend.device = 'mps'
open_backend._batch_chains = True
open_backend._batch_questions = True
open_backend._max_batch_size = 32
open_backend._max_batch_tokens = 32768
open_backend.tokenizer = tokenizer
open_backend.model = model
open_backend._last_logits = True
open_backend.eos = tokenizer.eos_token_id
open_backend.limit = min(model.config.max_position_embeddings, tokenizer.model_max_length)
open_client = LocalClient(model=MODEL, backend=open_backend)

simple_compiler = PromptCompiler(tokenizer, max_tokens=open_backend.limit)
simple_backend = SimpleBackend(model, max_batch_size=32, max_batch_tokens=32768)


def run_open(body):
    start = perf_counter()
    result = open_client.invoke(body)
    return perf_counter() - start, result, None


def run_simple(body):
    start = perf_counter()
    request = ClassifierRequest.model_validate(body)
    compiled = simple_compiler.compile(request)
    compiled_at = perf_counter()
    scored = simple_backend._score(compiled, threading.Event())
    scored_at = perf_counter()
    result = build_response(
        compiled.plan,
        scored.logits,
        input_tokens=unique_prompt_tokens([b.token_ids for b in compiled.branches]),
        output_tokens=0,
    )
    return perf_counter() - start, result, {
        **scored.metrics,
        'compile_seconds': compiled_at - start,
        'score_seconds': scored_at - compiled_at,
    }


from open_cricket.questionnaire import _prepare
from benchmark_answer_codes import CASES, OPTIONS

rows = []
for questions, candidates in [(1, 4), (4, 4), (16, 4), (1, 50), (1, 62), (1, 63), (1, 128), (1, 256)]:
    body = payload(questions)
    if candidates != 4:
        body['state'] = 'Route this request to candidate 0.'
        for question in body['questions'].values():
            question['criteria'] = {f'candidate {i}': None for i in range(candidates)}
    runners = {'open_cricket': run_open}
    rejection = None
    try:
        ClassifierRequest.model_validate(body)
        runners['simple_jev'] = run_simple
    except ValueError as exc:
        rejection = str(exc).splitlines()[1:3]
    for run in runners.values():
        run(body)
    times = {name: [] for name in runners}
    for repeat in range(args.repeats):
        order = list(runners) if repeat % 2 else list(reversed(runners))
        for name in order:
            elapsed, _, _ = runners[name](body)
            times[name].append(elapsed)
    q = next(iter(body['questions'].values()))
    prepared = _prepare(open_backend, body['state'], q['instructions'],
        [{'label': label, 'description': desc} for label, desc in q['criteria'].items()],
        'answer_codes', 1.0)
    rows.append({'questions': questions, 'candidates': candidates,
        'median_seconds': {k: statistics.median(v) for k,v in times.items()},
        'samples_seconds': times, 'simple_jev_rejection': rejection,
        'open_prompt_tokens': len(prepared['prompt']),
        'open_max_scored_tokens': max(map(len,prepared['paths'].values()))})
    print(f'Finished {questions} questions, {candidates} candidates', file=sys.stderr, flush=True)

accuracy = []
for state, expected in CASES:
    for reverse in (False, True):
        options = list(reversed(OPTIONS)) if reverse else OPTIONS
        body = payload(1)
        body['state'] = state
        body['questions']['route_0']['criteria'] = {x['label']:x['description'] for x in options}
        predictions = {}
        for name, run in [('open_cricket',run_open),('simple_jev',run_simple)]:
            _, response, _ = run(body)
            predictions[name] = response['answers']['route_0']['choice']
        accuracy.append({'state':state,'expected':expected,'reversed':reverse,**predictions})
print(json.dumps({'model': MODEL, 'device':'mps','dtype':str(next(model.parameters()).dtype),
    'torch':torch.__version__,'transformers':__import__('transformers').__version__,
    'repeats':args.repeats,'rows':rows,'smoke_accuracy':accuracy},indent=2))
