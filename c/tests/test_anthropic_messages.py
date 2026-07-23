"""Anthropic Messages API (#343): /v1/messages as a translation layer over the same
generation path the OpenAI endpoint uses.

The point of these tests is the *contract a real Anthropic client depends on* — Claude Code
is the reference client from the issue — not merely that the handler returns 200:
  - request translation: system prompt, content blocks, tool_use/tool_result round trips;
  - response shape: content blocks, stop_reason, usage with Anthropic's own field names;
  - the SSE event sequence, in order, with named events (a client that keys off
    `event:` names breaks on a data-only stream even if the JSON is right);
  - `x-api-key` auth, which is how Anthropic clients authenticate — Bearer keeps working;
  - the Anthropic error envelope, which is not the OpenAI one.
"""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import (APIServer, anthropic_to_openai, anthropic_tools, APIError)


class FakeEngine:
    """Emits whatever `script` says, so a test can drive tool syntax through the parser."""
    def __init__(self, script=("Hé", "llo"), length_limited=False):
        self.script = script
        self.length_limited = length_limited
        self.prompts = []

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None):
        self.prompts.append(prompt)
        for chunk in self.script:
            on_text(chunk)
        return {"prompt_tokens": 11, "completion_tokens": 3,
                "length_limited": self.length_limited}


class TranslationTest(unittest.TestCase):
    def test_system_and_blocks_become_openai_messages(self):
        messages = anthropic_to_openai({
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        })
        self.assertEqual(messages, [{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "Hi"}])

    def test_tool_use_and_result_round_trip(self):
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Rome"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
                {"type": "text", "text": "and tomorrow?"}]},
        ]})
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]),
                         {"city": "Rome"})
        # the tool result must precede the user's new question, or the model reads the
        # answer as arriving after a question it has not been asked yet
        self.assertEqual([m["role"] for m in messages[2:]], ["tool", "user"])
        self.assertEqual(messages[2]["content"], "18C")
        self.assertEqual(messages[3]["content"], "and tomorrow?")

    def test_tool_result_only_message_adds_no_empty_user_turn(self):
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "done"}]},
        ]})
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])

    def test_tools_and_choice_translate(self):
        tools, choice = anthropic_tools({
            "tools": [{"name": "f", "description": "d",
                       "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}],
            "tool_choice": {"type": "tool", "name": "f"}})
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["parameters"]["properties"], {"x": {"type": "string"}})
        self.assertEqual(choice, {"type": "function", "function": {"name": "f"}})
        self.assertEqual(anthropic_tools({"tool_choice": {"type": "any"}})[1], "required")
        self.assertEqual(anthropic_tools({"tool_choice": {"type": "auto"}})[1], "auto")

    def test_rejects_system_role_in_messages(self):
        with self.assertRaises(APIError) as caught:
            anthropic_to_openai({"messages": [{"role": "system", "content": "no"}]})
        self.assertIn("system", caught.exception.message)


class MessagesHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model", "secret", 64,
                                kv_slots=2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, body, headers=None, path="/v1/messages"):
        head = {"Content-Type": "application/json", "x-api-key": "secret"}
        head.update(headers or {})
        return urlopen(Request(self.base + path, data=json.dumps(body).encode(), headers=head),
                       timeout=3)

    def base_body(self, **extra):
        body = {"model": "test-model", "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hi"}]}
        body.update(extra)
        return body

    def test_message_response_shape(self):
        with self.post(self.base_body()) as response:
            payload = json.load(response)
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["role"], "assistant")
        self.assertEqual(payload["content"], [{"type": "text", "text": "Héllo"}])
        self.assertEqual(payload["stop_reason"], "end_turn")
        self.assertIsNone(payload["stop_sequence"])
        # Anthropic's usage field names, not OpenAI's prompt_tokens/completion_tokens
        self.assertEqual(payload["usage"], {"input_tokens": 11, "output_tokens": 3})
        self.assertTrue(payload["id"].startswith("msg_"))

    def test_x_api_key_and_bearer_both_authenticate(self):
        with self.post(self.base_body(), {"x-api-key": "secret"}) as response:
            self.assertEqual(response.status, 200)
        with self.post(self.base_body(), {"x-api-key": None and "" or "",
                                          "Authorization": "Bearer secret"}) as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(HTTPError) as caught:
            self.post(self.base_body(), {"x-api-key": "wrong"})
        self.assertEqual(caught.exception.code, 401)

    def test_error_envelope_is_anthropic_shaped(self):
        with self.assertRaises(HTTPError) as caught:
            self.post({"model": "test-model", "messages": [{"role": "user", "content": "x"}]})
        payload = json.load(caught.exception)
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("max_tokens", payload["error"]["message"])
        self.assertNotIn("param", payload)          # OpenAI's envelope must not leak here

    def test_max_tokens_maps_to_stop_reason(self):
        self.engine.length_limited = True
        with self.post(self.base_body()) as response:
            self.assertEqual(json.load(response)["stop_reason"], "max_tokens")

    def test_stream_emits_named_events_in_order(self):
        with self.post(self.base_body(stream=True)) as response:
            raw = response.read().decode()
        names = [line[len("event: "):] for line in raw.splitlines() if line.startswith("event: ")]
        self.assertEqual(names, ["message_start", "content_block_start", "content_block_delta",
                                 "content_block_delta", "content_block_stop", "message_delta",
                                 "message_stop"])
        payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                    if line.startswith("data: ")]
        text = "".join(p["delta"]["text"] for p in payloads
                       if p["type"] == "content_block_delta")
        self.assertEqual(text, "Héllo")
        self.assertEqual(payloads[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(payloads[-2]["usage"]["output_tokens"], 3)

    def test_tool_call_becomes_tool_use_block(self):
        self.engine.script = ("Sure. <tool_call>get_weather<arg_key>city</arg_key>"
                              "<arg_value>Rome</arg_value></tool_call>",)
        body = self.base_body(tools=[{"name": "get_weather", "description": "w",
                                      "input_schema": {"type": "object",
                                                       "properties": {"city": {"type": "string"}},
                                                       "required": ["city"]}}])
        with self.post(body) as response:
            payload = json.load(response)
        self.assertEqual(payload["stop_reason"], "tool_use")
        kinds = [block["type"] for block in payload["content"]]
        self.assertEqual(kinds, ["text", "tool_use"])
        call = payload["content"][1]
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["input"], {"city": "Rome"})
        self.assertTrue(call["id"])
        # the raw <tool_call> markup must never surface as visible text
        self.assertNotIn("tool_call", payload["content"][0]["text"])

    def test_streamed_tool_call_uses_input_json_delta(self):
        self.engine.script = ("<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",)
        body = self.base_body(stream=True, tools=[{"name": "f", "input_schema": {
            "type": "object", "properties": {"x": {"type": "string"}}}}])
        with self.post(body) as response:
            raw = response.read().decode()
        self.assertIn('"type":"input_json_delta"', raw)
        payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                    if line.startswith("data: ")]
        start = [p for p in payloads if p["type"] == "content_block_start"
                 and p["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(start), 1)
        self.assertEqual(start[0]["content_block"]["name"], "f")
        self.assertEqual(start[0]["index"], 1)      # block 0 is the text block
        self.assertEqual(payloads[-2]["delta"]["stop_reason"], "tool_use")

    def test_thinking_enabled_renders_reasoning_prompt(self):
        self.post(self.base_body(thinking={"type": "enabled"})).close()
        self.assertIn("Reasoning Effort", self.engine.prompts[-1])
        self.assertTrue(self.engine.prompts[-1].endswith("<|assistant|><think>"))

    def test_unsupported_fields_refuse_loudly(self):
        for field, value in (("stop_sequences", ["STOP"]), ("top_k", 40)):
            with self.assertRaises(HTTPError) as caught:
                self.post(self.base_body(**{field: value}))
            self.assertEqual(caught.exception.code, 400)
            self.assertIn(field, json.load(caught.exception)["error"]["message"])


if __name__ == "__main__":
    unittest.main()
