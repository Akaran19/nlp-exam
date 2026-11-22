"""all-MiniLM-L6-v2 sentence embeddings in plain numpy.

torch / onnxruntime have no wheels for this machine's Python (x86_64 under Rosetta), so the BERT forward pass is
written out here. Same weights, tokenizer, max length (256 word pieces), mean pooling and L2 normalisation as the
sentence-transformers model.
"""
import json

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from scipy.special import erf
from tokenizers import Tokenizer

REPO = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 256


class MiniLM:
    def __init__(self):
        self.w = {k.removeprefix("bert."): v.astype(np.float32)
                  for k, v in load_file(hf_hub_download(REPO, "model.safetensors")).items()}
        self.cfg = json.load(open(hf_hub_download(REPO, "config.json")))
        self.tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.no_padding()  # tokenizer.json pads to 128; single texts need no padding or mask

    @staticmethod
    def _ln(x, g, b, eps=1e-12):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def _lin(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _encode_one(self, text):
        ids = np.array(self.tok.encode(text).ids)
        w, n_heads = self.w, self.cfg["num_attention_heads"]
        x = (w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:len(ids)]
             + w["embeddings.token_type_embeddings.weight"][0])
        x = self._ln(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"])
        d = x.shape[1] // n_heads
        for i in range(self.cfg["num_hidden_layers"]):
            p = f"encoder.layer.{i}."
            q, k, v = (self._lin(x, p + f"attention.self.{n}").reshape(len(ids), n_heads, d).transpose(1, 0, 2)
                       for n in ("query", "key", "value"))
            a = q @ k.transpose(0, 2, 1) / np.sqrt(d)
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            ctx = (a @ v).transpose(1, 0, 2).reshape(len(ids), -1)
            x = self._ln(x + self._lin(ctx, p + "attention.output.dense"),
                         w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"])
            h = self._lin(x, p + "intermediate.dense")
            h = 0.5 * h * (1 + erf(h / np.sqrt(2)))
            x = self._ln(x + self._lin(h, p + "output.dense"),
                         w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"])
        e = x.mean(0)
        return e / np.linalg.norm(e)

    def encode(self, texts):
        return np.stack([self._encode_one(t) for t in texts])


if __name__ == "__main__":
    # sentence-transformers docs report 0.943 for this pair
    e = MiniLM().encode(["That is a happy person", "That is a very happy person", "Today is a sunny day"])
    print("happy/very happy:", round(float(e[0] @ e[1]), 4), " happy/sunny:", round(float(e[0] @ e[2]), 4))
