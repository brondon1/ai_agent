"""测试用的假模型响应和假向量模型，不需要 API 密钥和网络。"""

import hashlib
import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_agent.config import Config
from ai_agent.llm import LLM


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(id_, name, tool_input):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=tool_input)


def response(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


def answer(value):
    return response("end_turn", text(value))


def json_answer(data):
    return answer(json.dumps(data, ensure_ascii=False))


def fake_client(*responses):
    client = MagicMock()
    client.beta.messages.create.side_effect = list(responses)
    return client


def fake_llm(*responses, **config):
    client = fake_client(*responses)
    return LLM(Config(**config), client), client


class HashEmbedder:
    """按字符二元组哈希成向量：字面重合越多，余弦相似度越高。"""

    dim = 64

    def embed(self, texts):
        vectors = []
        for value in texts:
            vector = [0.0] * self.dim
            for i in range(len(value) - 1):
                digest = hashlib.md5(value[i : i + 2].encode()).digest()
                vector[digest[0] % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append([x / norm for x in vector])
        return vectors
