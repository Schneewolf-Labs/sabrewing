"""Gateway hardening ported from upstream colibri: slot admission, HTTP/1.1 framing and
keep-alive sync, stream cleanup, and engine-pipe failures."""
import http.client
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

from openai_server import (APIError, APIServer, ClientCancelled, Engine, GenerationScheduler,
                           _write_all)
from test_openai_server import FakeEngine, FakeProcess


class SchedulerAdmissionTest(unittest.TestCase):
    def test_any_slot_request_uses_a_slot_not_reserved_by_older_waiters(self):
        scheduler = GenerationScheduler(capacity=2)
        older = (object(), 0)                # an older waiter pinned to slot 0
        scheduler.queue.append(older)
        with patch.object(scheduler.condition, "wait", side_effect=AssertionError("idle slot 1")):
            with scheduler.admit() as (_, slot):
                self.assertEqual(slot, 1)
                self.assertEqual(list(scheduler.queue), [older])

    def test_older_any_slot_and_same_slot_waiters_keep_priority(self):
        for older_slot, requested in ((None, None), (None, 1), (0, 0)):
            with self.subTest(older_slot=older_slot, requested=requested):
                scheduler = GenerationScheduler(capacity=2)
                older = (object(), older_slot)
                scheduler.queue.append(older)
                with patch.object(scheduler.condition, "wait",
                                  side_effect=lambda _t: scheduler.queue.remove(older)) as wait:
                    with scheduler.admit(slot=requested) as (_, slot):
                        self.assertEqual(slot, 0 if requested is None else requested)
                wait.assert_called_once()

    def test_zero_queue_admits_free_slot_and_rejects_busy_pinned_slot(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=0.01, capacity=2)
        with scheduler.admit(slot=0):
            with self.assertRaises(APIError) as caught:
                with scheduler.admit(slot=0):
                    self.fail("busy pinned slot admitted")
            self.assertEqual(caught.exception.code, "queue_full")
            with scheduler.admit(slot=1) as (_, slot):   # a free slot is not "queueing"
                self.assertEqual(slot, 1)
        stats = scheduler.snapshot()
        self.assertEqual((stats["rejected"], stats["timed_out"], stats["admitted"]), (1, 0, 2))

    def test_cancelled_request_does_not_take_a_free_slot(self):
        scheduler = GenerationScheduler()
        with self.assertRaises(ClientCancelled):
            with scheduler.admit(lambda: True):
                self.fail("cancelled request admitted")
        stats = scheduler.snapshot()
        self.assertEqual((stats["admitted"], stats["cancelled"], stats["active"]), (0, 1, 0))

    def test_deadline_wins_over_a_slot_freed_too_late(self):
        for released_at, admitted in ((0.9, True), (1.1, False)):
            with self.subTest(released_at=released_at):
                scheduler = GenerationScheduler(queue_timeout=1)
                now = [0.0]
                with patch("openai_server.time.monotonic", side_effect=lambda: now[0]):
                    holder = scheduler.admit()
                    holder.__enter__()

                    def release(_timeout):
                        now[0] = released_at
                        holder.__exit__(None, None, None)
                    with patch.object(scheduler.condition, "wait", side_effect=release):
                        if admitted:
                            with scheduler.admit():
                                pass
                        else:
                            with self.assertRaises(APIError) as caught:
                                with scheduler.admit():
                                    pass
                            self.assertEqual(caught.exception.code, "queue_timeout")
                self.assertEqual(scheduler.snapshot()["admitted"], 2 if admitted else 1)

    def test_admitted_cancellation_is_not_counted_as_completed(self):
        scheduler = GenerationScheduler()
        with self.assertRaises(ClientCancelled):
            with scheduler.admit():
                raise ClientCancelled()
        stats = scheduler.snapshot()
        self.assertEqual((stats["completed"], stats["cancelled"], stats["active"]), (0, 1, 0))
        self.assertEqual(scheduler.free_slots, {0})


class _ExplodingEngine(FakeEngine):
    """Streams a little (committing the 200), then dies mid-generation."""
    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None):
        on_text("partial")
        raise RuntimeError("engine died mid-stream")


class FramingTest(unittest.TestCase):
    CHAT = {"model": "test-model", "messages": [{"role": "user", "content": "x"}]}

    def _server(self, engine=None, **kw):
        server = APIServer(("127.0.0.1", 0), engine or FakeEngine(), "test-model", **kw)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return server

    def _post(self, conn, body=None, headers=None, path="/v1/chat/completions"):
        head = {"Content-Type": "application/json"}
        head.update(headers or {})
        conn.request("POST", path, body=json.dumps(self.CHAT if body is None else body),
                     headers=head)
        response = conn.getresponse()
        return response.status, response.read()

    def _conn(self, server):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        self.addCleanup(conn.close)
        return conn

    def _raw(self, server, wire):
        """Byte-level exchange; returns everything until the server closes."""
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=3) as sock:
            sock.sendall(wire)
            out = bytearray()
            try:
                while chunk := sock.recv(65536):
                    out += chunk
            except socket.timeout:
                out += b"<STILL-OPEN>"           # connection was left reusable
            return out.decode("utf-8", "replace")

    def _wire(self, framing, payload=None, auth=b"", followup=True, path="/v1/chat/completions"):
        payload = (payload or json.dumps(self.CHAT)).encode()
        wire = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n".encode() + auth
                + b"Content-Type: application/json\r\n" + framing + b"\r\n" + payload)
        if followup:
            wire += b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        return wire

    # --- an early rejection must drain its body, or the next request line is garbage ---

    def test_rejected_auth_and_host_do_not_desync_the_next_request(self):
        server = self._server(api_key="secret")
        conn = self._conn(server)
        good = {"Authorization": "Bearer secret"}
        for _ in range(2):
            self.assertEqual(self._post(conn)[0], 401)
            self.assertEqual(self._post(conn, headers={"Host": "evil.example", **good})[0], 403)
            status, payload = self._post(conn, headers=good)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["object"], "chat.completion")
        self.assertIsNotNone(conn.sock, "plain JSON responses must stay persistent")

    def test_oversized_or_unparseable_length_closes_instead_of_desyncing(self):
        server = self._server()
        for length in (b"99999999", b"abc"):
            with self.subTest(length=length):
                raw = self._raw(server, self._wire(b"Content-Length: " + length + b"\r\n",
                                                   payload="{}", followup=False))
                self.assertIn(" 400 ", raw.splitlines()[0])
                self.assertNotIn("<STILL-OPEN>", raw)

    # --- ambiguous framing is refused before any engine work, and the socket closed ---

    def test_ambiguous_framing_is_refused(self):
        engine = FakeEngine()
        server = self._server(engine)
        n = len(json.dumps(self.CHAT))
        cases = {"te+cl": f"Content-Length: {n}\r\nTransfer-Encoding: chunked\r\n",
                 "dup cl": f"Content-Length: {n}\r\n" * 2,
                 "conflicting cl": f"Content-Length: {n}\r\nContent-Length: 1\r\n",
                 "signed cl": f"Content-Length: +{n}\r\n",
                 "underscore cl": f"Content-Length: {'_'.join(str(n))}\r\n"}
        for name, framing in cases.items():
            with self.subTest(name):
                raw = self._raw(server, self._wire(framing.encode()))
                self.assertTrue(raw.startswith("HTTP/1.1 400"), raw)
                self.assertEqual(raw.count("HTTP/1.1 "), 1, "connection was reused after it")
        self.assertEqual(engine.calls, [])

    def test_early_auth_error_closes_on_ambiguous_framing(self):
        server = self._server(api_key="secret")
        n = len(json.dumps(self.CHAT))
        raw = self._raw(server, self._wire(f"Content-Length: {n}\r\nContent-Length: 1\r\n".encode()))
        self.assertTrue(raw.startswith("HTTP/1.1 401"), raw)
        self.assertEqual(raw.count("HTTP/1.1 "), 1)

    def test_leading_zero_length_still_generates(self):
        engine = FakeEngine()
        server = self._server(engine)
        n = len(json.dumps(self.CHAT))
        raw = self._raw(server, self._wire(f"Content-Length: 000{n}\r\nConnection: close\r\n"
                                           .encode(), followup=False))
        self.assertTrue(raw.startswith("HTTP/1.1 200"), raw)
        self.assertEqual(len(engine.calls), 1)

    def test_dripping_client_runs_out_the_read_deadline(self):
        # The socket timeout renews on every byte; the cumulative deadline must not.
        server = self._server()
        with patch("openai_server.APIHandler.READ_DEADLINE", 0.5), \
                socket.create_connection(("127.0.0.1", server.server_port), timeout=3) as sock:
            started = time.monotonic()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n")
            closed = False
            for _ in range(20):                  # one header byte every 0.1 s, never finishing
                try:
                    sock.sendall(b"X")
                    sock.settimeout(0.1)
                    if sock.recv(4096) == b"":
                        closed = True
                        break
                except socket.timeout:
                    continue
                except OSError:
                    closed = True
                    break
            self.assertTrue(closed, "server kept reading a dripping request past its deadline")
            self.assertLess(time.monotonic() - started, 1.5)

    # --- SSE is close-framed, and a failure after the 200 must not splice a second response ---

    def test_streams_announce_close_framing(self):
        server = self._server()
        for path, body, tail in (
                ("/v1/chat/completions", dict(self.CHAT, stream=True), "data: [DONE]"),
                ("/v1/messages", dict(self.CHAT, stream=True, max_tokens=16), "message_stop")):
            with self.subTest(path=path):
                raw = self._raw(server, self._wire(f"Content-Length: {len(json.dumps(body))}\r\n"
                                                   .encode(), json.dumps(body), followup=False,
                                                   path=path))
                self.assertIn("connection: close", raw.split("\r\n\r\n", 1)[0].lower())
                self.assertIn(tail, raw)
                self.assertNotIn("<STILL-OPEN>", raw)

    def test_engine_failure_after_commit_does_not_splice_a_second_response(self):
        server = self._server(_ExplodingEngine())
        body = json.dumps(dict(self.CHAT, stream=True))
        raw = self._raw(server, self._wire(f"Content-Length: {len(body)}\r\n".encode(), body,
                                           followup=False))
        self.assertEqual(raw.count("HTTP/1."), 1, raw)
        self.assertIn("partial", raw)
        self.assertNotIn("<STILL-OPEN>", raw)

    def test_stream_exit_stops_keepalive_pump(self):
        original_thread = threading.Thread
        for path in ("/v1/chat/completions", "/v1/messages"):
            with self.subTest(path=path):
                pumps = []

                def thread_factory(*args, **kwargs):
                    thread = original_thread(*args, **kwargs)
                    if getattr(kwargs.get("target"), "__name__", "") in ("_keepalive", "keepalive"):
                        pumps.append(thread)
                    return thread
                server = self._server(_ExplodingEngine())
                with patch.object(threading, "Thread", side_effect=thread_factory):
                    status, _ = self._post(self._conn(server),
                                           dict(self.CHAT, stream=True, max_tokens=16), path=path)
                self.assertEqual(status, 200)
                self.assertEqual(len(pumps), 1)
                self.assertFalse(pumps[0].is_alive(), "keepalive survived an engine error")
                self.assertEqual(server.scheduler.snapshot()["active"], 0)

    # --- a lone surrogate is invalid client text, not an engine failure ---

    def test_unpaired_surrogate_is_a_400(self):
        engine = FakeEngine()
        server = self._server(engine)
        cut = "emoji \ud83d"
        for path, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": cut}]}),
                           ("/v1/completions", {"prompt": cut}),
                           ("/v1/messages", {"max_tokens": 4,
                                             "messages": [{"role": "user", "content": cut}]})):
            with self.subTest(path=path):
                status, payload = self._post(self._conn(server), dict(body, model="test-model"),
                                             path=path)
                self.assertEqual(status, 400, payload)
        self.assertEqual(engine.calls, [])
        status, _ = self._post(self._conn(server), {"model": "test-model", "prompt": "😀"},
                               path="/v1/completions")   # a paired escape is a real emoji
        self.assertEqual(status, 200)


class EnginePipeTest(unittest.TestCase):
    def _engine(self, respond):
        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        self.addCleanup(engine.close)
        return engine, process

    def test_cancels_during_prefill_before_any_frame(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]          # silent: the engine is "prefilling"
            elif fields[0] == b"CANCEL":
                process.stdout.feed(b"ERROR " + request_id + b" CANCELLED\n")

        engine, process = self._engine(respond)
        flag = {"gone": False}
        outcome = []

        def generate():
            try:
                engine.generate("hello", 8, 0.7, 0.9, lambda _: None, cancelled=lambda: flag["gone"])
            except ClientCancelled:
                outcome.append("cancelled")
        thread = threading.Thread(target=generate)
        thread.start()
        for _ in range(200):
            if process.writes:
                break
            time.sleep(0.01)
        flag["gone"] = True
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, ["cancelled"])
        self.assertEqual(process.writes[-1].split(), [b"CANCEL", request_id])

    def test_dead_engine_stdin_is_a_named_error_not_a_client_hangup(self):
        def respond(_process, _frame):
            raise BrokenPipeError(32, "Broken pipe")

        engine, _ = self._engine(respond)
        with self.assertRaisesRegex(RuntimeError, "failed to write SUBMIT") as caught:
            engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
        self.assertNotIsInstance(caught.exception, ConnectionError)
        self.assertEqual(engine.pending, {})

    def test_short_writes_are_completed_and_no_progress_fails(self):
        class Trickle:
            def __init__(self, step):
                self.step, self.out = step, bytearray()
            def write(self, data):
                self.out += data[:self.step]
                return self.step and min(self.step, len(data))
            def flush(self):
                pass
        frame = b"SUBMIT 1 0 5 4\nhello\n"
        stream = Trickle(3)
        _write_all(stream, frame, "SUBMIT")
        self.assertEqual(bytes(stream.out), frame)
        with self.assertRaisesRegex(RuntimeError, "took 0 of"):
            _write_all(Trickle(0), frame, "SUBMIT")

if __name__ == "__main__":
    unittest.main()
