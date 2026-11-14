"""Real local HTTP requests: inactivity timeouts cannot detect trickle bodies."""

from concurrent.futures import ThreadPoolExecutor
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sqlite3
import threading
import time
import unittest
from unittest import mock

import httpx
from openai import OpenAI, APITimeoutError

from app import endpoints
from app.request_deadline import DeadlineTransport, RequestDeadlineExceeded, deadline_reason


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, headers=None):
        self.send_response(200)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def trickle(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        try:
            for _ in range(120):
                self.wfile.write(b' ')
                self.wfile.flush()
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.server.disconnected.set()

    def do_GET(self):
        try:
            if self.path == '/trickle':
                self.trickle()
            elif self.path == '/headers':
                time.sleep(0.7)
                self.reply(b'{}')
            elif self.path.startswith('/redirect/'):
                step = int(self.path.rsplit('/', 1)[1])
                time.sleep(0.12)
                if step < 4:
                    self.send_response(302)
                    self.send_header('Location', f'/redirect/{step + 1}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                else:
                    self.reply(b'{}')
            elif self.path == '/gzip':
                self.reply(gzip.compress(b'{"ok": true}'), {'Content-Encoding': 'gzip'})
            elif self.path == '/private-redirect':
                self.send_response(302)
                self.send_header('Location', 'http://127.0.0.1:9/secret')
                self.send_header('Content-Length', '0')
                self.end_headers()
            else:
                self.reply(b'{"ok": true}')
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.count_lock:
            self.server.calls += 1
            call = self.server.calls
        if body.get('model') == 'slow' or (body.get('model') == 'retry' and call == 1):
            self.trickle()
        else:
            response = {'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                        'choices': [{'index': 0, 'finish_reason': 'stop',
                                     'message': {'role': 'assistant', 'content': 'done'}}],
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
            self.reply(json.dumps(response).encode(), {'Content-Type': 'application/json'})


class RequestDeadlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.daemon_threads = True
        cls.server.disconnected = threading.Event()
        cls.server.count_lock = threading.Lock()
        cls.server.calls = 0
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def client(self, seconds=0.3, validate=lambda url: None):
        return httpx.Client(transport=DeadlineTransport(seconds, validate),
                            timeout=2, follow_redirects=True, trust_env=False)

    def test_trickling_body_is_cancelled_and_socket_closes(self):
        self.server.disconnected.clear()
        with self.client() as client:
            start = time.monotonic()
            with self.assertRaises(RequestDeadlineExceeded):
                client.get(self.url + '/trickle')
            self.assertLess(time.monotonic() - start, 1.2)
            self.assertTrue(self.server.disconnected.wait(1.5), 'Timed-out HTTP operation must close its socket')
            self.assertEqual(client.get(self.url + '/fast').json(), {'ok': True})

    def test_response_headers_are_also_inside_deadline(self):
        with self.client() as client:
            start = time.monotonic()
            with self.assertRaises(RequestDeadlineExceeded):
                client.get(self.url + '/headers')
            self.assertLess(time.monotonic() - start, 0.65)

    def test_redirects_share_one_deadline(self):
        with self.client() as client:
            start = time.monotonic()
            with self.assertRaises(RequestDeadlineExceeded):
                client.get(self.url + '/redirect/1')
            self.assertLess(time.monotonic() - start, 0.65)

    def test_compressed_response_is_decoded_once(self):
        with self.client() as client:
            self.assertEqual(client.get(self.url + '/gzip').json(), {'ok': True})

    def test_concurrent_requests_do_not_block_one_another(self):
        with self.client(0.6) as client, ThreadPoolExecutor(max_workers=2) as executor:
            slow = executor.submit(client.get, self.url + '/trickle')
            fast = executor.submit(client.get, self.url + '/fast')
            self.assertEqual(fast.result(timeout=0.4).json(), {'ok': True})
            with self.assertRaises(RequestDeadlineExceeded):
                slow.result(timeout=1.5)

    def test_validation_is_inside_deadline(self):
        release = threading.Event()
        def validate(url):
            release.wait(2)
        try:
            with self.client(validate=validate) as client:
                with self.assertRaises(RequestDeadlineExceeded):
                    client.get(self.url + '/fast')
        finally:
            release.set()

    def test_redirect_target_is_validated(self):
        checked = []
        def validate(url):
            checked.append(url)
            if ':9/' in url:
                raise endpoints.EndpointNotAllowed('Private target')
        with self.client(validate=validate) as client:
            with self.assertRaises(endpoints.EndpointNotAllowed):
                client.get(self.url + '/private-redirect')
        self.assertEqual(len(checked), 2)

    def test_openai_sdk_preserves_deadline_cause(self):
        with mock.patch.object(endpoints, 'ALLOW_PRIVATE', True), OpenAI(
            base_url=self.url + '/v1', api_key='test', max_retries=0,
            http_client=endpoints.http_client(total_timeout=0.3)
        ) as client:
            with self.assertRaises(APITimeoutError) as raised:
                client.chat.completions.create(model='slow', messages=[{'role': 'user', 'content': 'test'}])
            self.assertIn('Total request deadline of 0.3s', deadline_reason(raised.exception))

    def test_runner_retries_with_fresh_deadline_and_logs_reason(self):
        from app.disambig.runner import DisambigConfig, LLMClient
        logs = []
        with self.server.count_lock:
            self.server.calls = 0
        config = DisambigConfig(llm_url=self.url + '/v1', api_key='test', model='retry',
                                topic='test', seed_concept='test', request_timeout_seconds=0.3, pacing=False)
        with mock.patch.object(endpoints, 'ALLOW_PRIVATE', True), \
             mock.patch('app.reasoning.capability_for', return_value=None), \
             mock.patch.object(LLMClient, '_retry_delay', return_value=0):
            client = LLMClient(config, threading.Event(), lambda message, level: logs.append(message))
            try:
                self.assertEqual(client.chat('elicitation', '', 'test', max_retries=1), 'done')
            finally:
                for endpoint in client.endpoints.values():
                    endpoint.client.close()
        self.assertEqual(self.server.calls, 2)
        self.assertTrue(any('Total request deadline' in line and 'retry 1 of 1' in line for line in logs))

    def test_exhausted_retry_clears_heartbeat(self):
        from app.disambig.runner import DisambigConfig, LLMClient
        from app.heartbeat import Heartbeat
        logs = []
        beat = Heartbeat(lambda *args: None, lambda: (None, None), threading.Event())
        config = DisambigConfig(llm_url=self.url + '/v1', api_key='test', model='slow',
                                topic='test', seed_concept='test', request_timeout_seconds=0.3, pacing=False)
        with mock.patch.object(endpoints, 'ALLOW_PRIVATE', True), \
             mock.patch('app.reasoning.capability_for', return_value=None):
            client = LLMClient(config, threading.Event(), lambda message, level: logs.append(message), heartbeat=beat)
            try:
                with self.assertRaises(APITimeoutError):
                    client.chat('elicitation', '', 'test', max_retries=0)
                self.assertEqual(beat._requests, {})
            finally:
                for endpoint in client.endpoints.values():
                    endpoint.client.close()
        self.assertTrue(any('Total request deadline' in line and 'no retries left' in line for line in logs))

    def test_embeddings_use_their_total_deadline(self):
        from app.disambig import embeddings
        with sqlite3.connect(':memory:') as db, \
             mock.patch.object(endpoints, 'ALLOW_PRIVATE', True), \
             mock.patch.object(embeddings, 'REQUEST_TIMEOUT_SECONDS', 0.3):
            client = embeddings.EmbeddingClient(self.url + '/v1', 'test', 'slow', db,
                                                threading.Event(), lambda *args: None)
            try:
                start = time.monotonic()
                with self.assertRaises(APITimeoutError) as raised:
                    client._request(['test'], max_retries=0)
                self.assertLess(time.monotonic() - start, 1.2)
                self.assertIn('Total request deadline of 0.3s', deadline_reason(raised.exception))
            finally:
                client.client.close()
