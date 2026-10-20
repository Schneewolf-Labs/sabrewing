"""GLM-5.3-Flash in the gateway: the chat template and the always-on reasoning split.

The golden prompts below were produced by upstream colibri's render_chat_glm53, which
upstream pins byte for byte against the checkpoint's chat_template.jinja (sha256
34d5ee66...3891, tests/glm53_chat_template_harness.py there). When this port landed it
matched that renderer on 54/54 renders (thinking on/off, every effort level, tools,
calls and results, history with reasoning).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import openai_server as srv  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "meteo", "description": "Il tempo",
    "parameters": {"type": "object", "properties": {"citta": {"type": "string"}},
                   "required": ["citta"]}}}]


class Glm53Template(unittest.TestCase):
    def test_plain_turn_defaults_to_low_effort_and_opens_think(self):
        self.assertEqual(srv.render_chat_glm53([{"role": "user", "content": "ciao"}]),
                         "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>ciao<|assistant|><think>")

    def test_tools_call_and_result(self):
        msgs = [{"role": "user", "content": "che tempo fa a Roma?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": "meteo", "arguments": {"citta": "Roma"}}}]},
                {"role": "tool", "content": "sereno, 24 gradi"}]
        want = ('[gMASK]<sop><|system|>Reasoning Effort: High<|system|>\n# Tools\n\nYou may call one or '
                'more functions to assist with the user query.\n\nYou are provided with function '
                'signatures within <tools></tools> XML tags:\n<tools>\n\n{"name": "meteo", "description": '
                '"Il tempo", "parameters": {"type": "object", "properties": {"citta": {"type": "string"}}, '
                '"required": ["citta"]}}\n\n\n</tools>\n\nFor each function call, output the function name '
                'and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>'
                '{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>'
                '<arg_value>{arg-value-2}</arg_value>...</tool_call><|user|>che tempo fa a Roma?'
                '<|assistant|><think></think><tool_call>meteo<arg_key>citta</arg_key><arg_value>Roma'
                '</arg_value></tool_call>\n<|observation|><tool_response>sereno, 24 gradi'
                '</tool_response><|assistant|><think>')
        self.assertEqual(srv.render_chat_glm53(msgs, enable_thinking=True, reasoning_effort="high",
                                               tools=TOOLS), want)

    def test_effort_ladder(self):
        for effort, line in (("minimal", "Low"), ("low", "Low"), ("medium", "High"),
                             ("high", "High"), ("xhigh", "Max")):
            p = srv.render_chat_glm53([{"role": "user", "content": "x"}], True, effort)
            self.assertIn(f"Reasoning Effort: {line}<|user|>", p)
        p = srv.render_chat_glm53([{"role": "user", "content": "x"}], True, None)
        self.assertIn("Reasoning Effort: Max<|user|>", p)

    def test_reply_splits_reasoning_from_content(self):
        # the prompt opened <think>, so the reply starts inside the reasoning block
        reasoning, answer = srv.split_thinking_reply("weigh it up</think>Rome is sunny.")
        self.assertEqual((reasoning, answer), ("weigh it up", "Rome is sunny."))

    def test_stream_split_never_leaks_markers(self):
        thinking, text = [], []
        split = srv.ThinkingStreamSplit(thinking.append, text.append)
        for chunk in ("wei", "gh</th", "ink>Ro", "me"):
            split.feed(chunk)
        split.close()
        self.assertEqual(("".join(thinking), "".join(text)), ("weigh", "Rome"))


if __name__ == "__main__":
    unittest.main()
