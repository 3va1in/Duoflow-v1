import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from astra_planner import AstraPlannerBackend, AstraPlannerError, AstraAPIError


class Response(io.BytesIO):
    status = 200


def response(skill_id='pick'):
    return {'status': 'completed', 'usage': {'input_tokens': 10, 'output_tokens': 5},
            'output': [{'type': 'message', 'status': 'completed', 'content': [
                {'type': 'output_text', 'text': json.dumps({'skill_id': skill_id, 'status': 'continue', 'confidence': 0.6})}]}]}


class AstraPlannerTests(unittest.TestCase):
    def call(self, planner):
        return planner.rank({'head': np.zeros((3, 4, 3), dtype=np.uint8)}, np.zeros(14),
                            'Put the can in the basket.',
                            [{'skill_id': 'pick', 'execution_prompt': 'Pick up the can.'}],
                            [{'text': 'Previous episode', 'outcome': 'failure'}], history=[])

    def test_request_decision_and_audit_do_not_require_native_features(self):
        with tempfile.TemporaryDirectory() as directory:
            planner = AstraPlannerBackend(api_key='test-secret', audit_dir=directory)
            with patch('astra_planner.urlopen', return_value=Response(json.dumps(response()).encode())) as transport:
                result = self.call(planner)
            request = transport.call_args.args[0]
            sent = json.loads(request.data)
            self.assertEqual(sent['model'], 'gpt-6-astra')
            self.assertEqual(sent['reasoning'], {'effort': 'low'})
            self.assertTrue(sent['text']['format']['strict'])
            self.assertEqual(sent['text']['format']['schema']['properties']['skill_id']['enum'], ['pick'])
            self.assertTrue(sent['input'][0]['content'][-1]['image_url'].startswith('data:image/png;base64,'))
            self.assertEqual(result['skill_id'], 'pick')
            self.assertEqual(result['scores'], [])
            self.assertEqual(result['selection_distribution'], 'api_structured_decision')
            self.assertFalse(hasattr(planner, 'last_candidate_features'))
            audit = Path(directory) / 'decision_000001'
            request_text = (audit / 'request.json').read_text()
            self.assertNotIn('base64', request_text)
            self.assertNotIn('test-secret', request_text)
            self.assertEqual(json.loads((audit / 'response.json').read_text())['usage']['output_tokens'], 5)

    def test_failures_do_not_retry_and_keep_error_evidence(self):
        incomplete = response()
        incomplete['status'] = 'incomplete'
        refusal = response()
        refusal['output'][0]['content'] = [{'type': 'refusal', 'refusal': 'Cannot select.'}]
        bad_json = response()
        bad_json['output'][0]['content'][0]['text'] = '{'
        nan_confidence = response()
        nan_confidence['output'][0]['content'][0]['text'] = '{"skill_id":"pick","status":"continue","confidence":NaN}'
        cases = [incomplete, refusal, bad_json, nan_confidence, response('outside')]
        with tempfile.TemporaryDirectory() as directory:
            planner = AstraPlannerBackend(api_key='test-secret', audit_dir=directory)
            for index, body in enumerate(cases, 1):
                with self.subTest(index=index), patch('astra_planner.urlopen', return_value=Response(json.dumps(body).encode())) as transport:
                    with self.assertRaises(AstraPlannerError):
                        self.call(planner)
                    transport.assert_called_once()
                    self.assertTrue((Path(directory) / f'decision_{index:06d}' / 'error.json').is_file())
            error = HTTPError('https://api.openai.com/v1/responses', 429, 'rate limit', {}, io.BytesIO(b'{"error":"test-secret"}'))
            with patch('astra_planner.urlopen', side_effect=error) as transport:
                with self.assertRaises(AstraAPIError) as caught:
                    self.call(planner)
                self.assertEqual(caught.exception.payload['http_status'], 429)
                self.assertEqual(caught.exception.payload['code'], 'rate_limit_exceeded')
                transport.assert_called_once()
            self.assertNotIn('test-secret', (Path(directory) / 'decision_000006' / 'response.json').read_text())

    def test_credentials_and_invalid_rgb_are_local_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'credential'
            for value in ['test-secret', 'OPENAI_API_KEY="test-secret"', '{"api_key":"test-secret"}']:
                path.write_text(value)
                planner = AstraPlannerBackend(credential_file=path)
                with patch('astra_planner.urlopen') as transport:
                    with self.assertRaisesRegex(AstraPlannerError, 'uint8 HWC RGB'):
                        planner.rank({'head': np.zeros((2, 2, 3))}, [0], 'Task',
                                     [{'skill_id': 'pick', 'execution_prompt': 'Pick.'}], [])
                    transport.assert_not_called()
            with patch.dict('os.environ', {}, clear=True):
                with self.assertRaisesRegex(AstraPlannerError, '缺少有效 API 密钥'):
                    AstraPlannerBackend()


if __name__ == '__main__':
    unittest.main()
