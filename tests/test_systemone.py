"""Jev HTTP and official SDK contract tests; no remote inference or model download."""
import importlib.util
import json
import os
import unittest
from unittest.mock import patch

HAS_SERVER = all(importlib.util.find_spec(m) for m in ('fastapi', 'httpx', 'langchain_core'))


class Classifier:
    def __init__(self):
        self.calls = []

    async def score(self, value):
        self.calls.append(value)
        labels = [x if isinstance(x, str) else x['label'] for x in value['options']]
        weights = range(1, len(labels) + 1)
        return {'options': [{'label': label, 'probability': weight / sum(weights)}
                            for label, weight in zip(labels, weights)], 'prompt_tokens': 17}

    async def ainvoke(self, value):
        from labeljudge.systemone import SystemOneRequest, classification_input, format_response
        request = SystemOneRequest.model_validate(value)
        results = [await self.score(classification_input(request.state, question))
                   for question in request.questions.values()]
        return format_response(request, request.model, results).model_dump(mode="json")


def payload():
    return {'model': 'local-checkpoint', 'state': {'ticket': 'Charged twice', 'history': [None, 2, True]},
            'questions': {
                'department': {'type': 'choice', 'instructions': {'task': 'Choose a department'},
                               'criteria': {'billing': {'covers': ['refunds']}, 'other': None}},
                'urgency': {'type': 'score', 'criteria': ['Can wait', {'deadline': 'today'}, ['Immediate']]},
                'billing': {'type': 'noul', 'instructions': 'Is this about billing?',
                            'criteria': {'true': 'Payment issue', 'false': ['Other issue']}}
            }}


@unittest.skipUnless(HAS_SERVER, 'Install .[test]')
class SystemOneTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from labeljudge.server import create_app
        self.classifier = Classifier()
        self.environment = patch.dict(os.environ, {'LABELJUDGE_API_KEY': 'local-test'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client = TestClient(create_app(self.classifier, model_name='local-checkpoint'))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {'Authorization': 'Bearer local-test'}

    def post(self, body):
        return self.client.post('/v1/systemone', json=body, headers=self.headers)

    def test_mixed_request_response_and_structured_content(self):
        response = self.post(payload())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(set(result), {'model', 'answers', 'usage'})
        self.assertEqual(result['model'], 'local-checkpoint')
        self.assertEqual(result['usage'], {'input_tokens': 51, 'output_tokens': 0})
        answers = result['answers']
        self.assertEqual(set(answers), set(payload()['questions']))
        self.assertEqual(answers['department']['choice'], 'other')
        self.assertEqual(answers['department']['probabilities'], {'billing': 1/3, 'other': 2/3})
        self.assertAlmostEqual(answers['urgency']['score'], 4/3)
        self.assertEqual(answers['urgency']['legend']['1'], {'deadline': 'today'})
        self.assertEqual(answers['billing'], {'type': 'noul', 'noul': 1/3})
        self.assertTrue(response.headers['x-request-id'])
        self.assertEqual(response.headers['x-labeljudge-confidence-method'], 'normalized-entropy')
        self.assertEqual(json.loads(self.classifier.calls[0]['message']), payload()['state'])
        self.assertEqual(json.loads(self.classifier.calls[0]['question']), {'task': 'Choose a department'})
        description = self.classifier.calls[0]['options'][0]['description']
        self.assertEqual(json.loads(description), {'covers': ['refunds']})

    def test_auth_models_aliases_and_request_ids(self):
        self.assertEqual(self.client.post('/v1/systemone', json=payload()).status_code, 401)
        self.assertEqual(self.client.get('/v1/models').status_code, 401)
        models = self.client.get('/v1/models', headers=self.headers)
        self.assertEqual(models.status_code, 200)
        names = [m['name'] for m in models.json()['models']]
        self.assertIn('local-checkpoint', names)
        self.assertIn('local-checkpoint', names)
        self.assertFalse(any('jev' in name for name in names))
        self.assertNotIn('x-typesafe-request-id', models.headers)
        for name in ('labeljudge', 'jev-latest', 'jev-preview', 'jev', 'jev-1.13.0'):
            self.assertEqual(self.post({**payload(), 'model': name}).status_code, 422)
        for name in names:
            result = self.post({**payload(), 'model': name})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['model'], 'local-checkpoint')
        self.assertEqual(self.post({**payload(), 'model': 'typo'}).status_code, 422)
        self.assertNotEqual(self.post(payload()).headers['x-request-id'],
                            self.post(payload()).headers['x-request-id'])

    def test_validation_before_inference(self):
        cases = [
            {**payload(), 'state': None}, {**payload(), 'state': 4},
            {**payload(), 'questions': {}},
            {**payload(), 'questions': {'q': {'type': 'unknown'}}},
            {**payload(), 'questions': {'q': {'type': 'choice', 'criteria': {}}}},
            {**payload(), 'questions': {'q': {'type': 'choice', 'criteria': {' ': None}}}},
            {**payload(), 'questions': {'q': {'type': 'score', 'criteria': []}}},
            {**payload(), 'questions': {'q': {'type': 'score', 'criteria': ['x'] * 11}}},
            {**payload(), 'questions': {'q': {'type': 'noul', 'criteria': {'yes': 'x'}}}},
            {**payload(), 'questions': {'q': {'type': 'noul', 'instructions': 9}}},
            {key: value for key, value in payload().items() if key != 'model'},
        ]
        for body in cases:
            with self.subTest(body=body):
                response = self.post(body)
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.classifier.calls, [])
        invalid_json = self.client.post('/v1/systemone', content='{', headers=self.headers)
        self.assertEqual(invalid_json.status_code, 422)

    def test_optional_instructions_and_noul_criteria(self):
        questions = {'a': {'type': 'choice', 'criteria': {'yes': None}},
                     'b': {'type': 'score', 'instructions': None, 'criteria': ['Only level']},
                     'c': {'type': 'noul'},
                     'd': {'type': 'noul', 'criteria': {'true': None}}}
        response = self.post({'model': 'local-checkpoint', 'state': [], 'questions': questions})
        self.assertEqual(response.status_code, 200, response.text)
        answers = response.json()['answers']
        self.assertEqual(answers['a']['confidence'], 1.0)
        self.assertEqual(answers['b']['score'], 0.0)
        self.assertEqual(answers['c']['noul'], 1/3)
        self.assertEqual(answers['c'], answers['d'])

    def test_normalized_entropy_boundaries(self):
        from labeljudge.systemone import confidence
        self.assertEqual(confidence({'a': .5, 'b': .5}), 0.0)
        self.assertEqual(confidence({'a': 1, 'b': 0}), 1.0)
        self.assertTrue(0 < confidence({'a': .8, 'b': .2}) < 1)

    def test_questionnaire_backend(self):
        from fastapi.testclient import TestClient
        from labeljudge.integrations import as_runnable
        from labeljudge.server import create_app

        class Tokens:
            def prompt_ids(self, system, form):
                return [99]

            def answer_ids(self, answer):
                return list(answer.encode()) + [0]

            def next_logprobs(self, prompt, prefix, allowed):
                return {token: -5.6 for token in allowed}

        with TestClient(create_app(as_runnable(Tokens(), model="local-checkpoint"), model_name="local-checkpoint")) as client:
            response = client.post('/v1/systemone', json=payload(), headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['usage']['input_tokens'], 3)

    def sdk_transport(self):
        import httpx2

        def forward(request):
            response = self.client.request(request.method, request.url.path,
                                           content=request.content, headers=dict(request.headers))
            return httpx2.Response(response.status_code, content=response.content,
                                   headers=dict(response.headers), request=request)

        return httpx2.MockTransport(forward)

    @unittest.skipUnless(importlib.util.find_spec('typesafe_sdk'), 'Install .[test] with typesafe-sdk')
    def test_official_sdk_round_trip(self):
        from typesafe_sdk import Choice, Noul, Score, TypeSafeClient, TypeSafeAPIError

        with TypeSafeClient(base_url='http://testserver', api_key='local-test', model='local-checkpoint',
                            transport=self.sdk_transport()) as sdk:
            result = sdk.system_one(state=payload()['state'], questions={
                'department': Choice(criteria={'billing': None, 'other': None}),
                'urgency': Score(criteria=['Low', {'severity': 'High'}]),
                'billing': Noul(instructions='Billing issue?')})
            self.assertEqual(result.choices['department'].choice, 'other')
            self.assertEqual(result.scores['urgency'].legend[1], {'severity': 'High'})
            self.assertAlmostEqual(result.scores['urgency'].score, 2/3)
            self.assertEqual(result.nouls['billing'].noul, 1/3)
            self.assertTrue(result.raw_http_response.headers['x-request-id'])
            self.assertEqual(result.usage.input_tokens, 51)
            self.assertIn('local-checkpoint', [m.name for m in sdk.models.list().models])
            with self.assertRaises(TypeSafeAPIError) as error:
                sdk.system_one('test', {'q': Noul()}, model='unknown')
            self.assertEqual(error.exception.status, 422)

    @unittest.skipUnless(importlib.util.find_spec('typesafe_sdk'), 'Install .[test] with typesafe-sdk')
    def test_official_async_sdk_and_environment_configuration(self):
        import asyncio
        from typesafe_sdk import AsyncTypeSafeClient, Noul

        async def run():
            async with AsyncTypeSafeClient(transport=self.sdk_transport()) as sdk:
                result = await sdk.system_one('A test message', {'q': Noul()})
                self.assertEqual(result.nouls['q'].noul, 1/3)
                self.assertTrue(result.raw_http_response.headers['x-request-id'])
                models = await sdk.models.list()
                self.assertIn('local-checkpoint', [m.name for m in models.models])

        with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'local-test',
                                    'TYPESAFE_BASE_URL': 'http://testserver',
                                    'TYPESAFE_DEFAULT_MODEL': 'local-checkpoint'}):
            asyncio.run(run())

    def test_question_names_and_siblings_do_not_change_inference_input(self):
        body = payload()
        self.post(body)
        initial = self.classifier.calls[0]
        self.classifier.calls.clear()
        self.post({**body, 'questions': {'arbitrary-identifier': body['questions']['department']}})
        self.assertEqual(self.classifier.calls, [initial])

    def test_documented_limits(self):
        for count, status in [(255, 200), (256, 422)]:
            body = {'model': 'local-checkpoint', 'state': '', 'questions': {
                'q': {'type': 'choice', 'criteria': {str(i): None for i in range(count)}}}}
            self.assertEqual(self.post(body).status_code, status)

    def test_backend_errors_and_missing_usage(self):
        from unittest.mock import AsyncMock
        self.classifier.score = AsyncMock(side_effect=ValueError('Input exceeds context limit'))
        self.assertEqual(self.post(payload()).status_code, 422)
        self.classifier.score = AsyncMock(return_value={
            'options': [{'label': 'true', 'probability': 0.7}, {'label': 'false', 'probability': 0.3}]})
        body = {'model': 'local-checkpoint', 'state': 'x', 'questions': {'q': {'type': 'noul'}}}
        response = self.post(body)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['usage']['input_tokens'])
        self.classifier.score = AsyncMock(return_value={
            'options': [{'label': 'true', 'probability': 1.0}]})
        self.assertEqual(self.post(body).status_code, 422)
