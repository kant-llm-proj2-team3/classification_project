"""
CLM 판단 헤드 (state head + action head + logit_scale)

  HeadModel      : 공식 체크포인트(.pt)나 우리가 저장한 헤드(head.safetensors)를 불러와 선택지별 점수를 계산한다.
  infonce_loss   : CLM 공식 choice 학습의 양방향 InfoNCE (CLM 저장소 train/finetune.py의 run_choice와 같은 식)
  ClmClassifier  : best_model 폴더의 헤드와 인코더로 문장을 분류한다. (test 평가와 서버에서 쓴다)

점수 = exp(logit_scale)(최대 100) x cos(state_head(문장 + 질문), action_head(선택지 설명)),
선택지끼리 softmax한 값이 확률이다. 헤드는 학습에 쓴 인코더(GGUF, 풀링, 길이)와 짝이 맞을 때만 의미가 있어서,
저장할 때 encoder.json을 함께 두고 불러올 때 현재 인코더와 같은지 확인한다.
"""

import functools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

import clm_settings as S
from util.clm_encoder import HIDDEN, load_clm

HEAD_FILE = "head.safetensors"
HEAD_CONFIG_FILE = "head_config.json"
ENCODER_FILE = "encoder.json"
QUESTIONS_FILE = "questions.json"
ARCH_KEYS = ("width", "depth", "projection_dim", "activation", "layernorm", "residual", "hidden_size")


@functools.lru_cache(maxsize=1)
def base_head_path() -> Path:
    """공식 판단 헤드(CLM_v0.1-8B.pt)의 경로. HF 캐시에 없으면 내려받는다. (약 72MB)"""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(S.BASE_HEAD["repo"], S.BASE_HEAD["file"]))


class HeadModel:
    def __init__(self, arch: dict, device="cpu"):
        make_head = load_clm().heads.make_head
        self.arch = {key: arch[key] for key in ARCH_KEYS}
        self.device = torch.device(device)
        kwargs = {"width": arch["width"], "depth": arch["depth"], "proj": arch["projection_dim"],
                  "activation": arch["activation"], "layernorm": arch["layernorm"], "residual": arch["residual"],
                  "hidden": arch["hidden_size"]}
        self.state_head = make_head(**kwargs).to(self.device)
        self.action_head = make_head(**kwargs).to(self.device)
        self.logit_scale = nn.Parameter(torch.zeros((), device=self.device))

    # ── 불러오기 · 저장 ──
    @classmethod
    def from_checkpoint(cls, path: Path, device="cpu") -> "HeadModel":
        """공식 .pt 체크포인트. weights_only=True로 읽어서 파일 안의 코드가 실행되지 않는다."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        cfg = checkpoint["cfg"]
        arch = {"width": cfg["width"], "depth": cfg["depth"],
                "projection_dim": checkpoint.get("projection_dim", cfg.get("projection_dim", 512)),
                "activation": cfg.get("activation", "gelu"), "layernorm": cfg.get("layernorm", False),
                "residual": cfg.get("residual", False), "hidden_size": cfg.get("hidden_size", HIDDEN)}
        model = cls(arch, device)
        model.state_head.load_state_dict(checkpoint["state_head"])
        model.action_head.load_state_dict(checkpoint["action_head"])
        with torch.no_grad():
            model.logit_scale.copy_(torch.as_tensor(checkpoint["logit_scale"]).float())
        return model

    @classmethod
    def load(cls, directory: Path, device="cpu") -> "HeadModel":
        directory = Path(directory)
        arch = json.loads((directory / HEAD_CONFIG_FILE).read_text(encoding="utf-8"))
        model = cls(arch, device)
        tensors = load_file(directory / HEAD_FILE)
        for name, head in (("state_head", model.state_head), ("action_head", model.action_head)):
            head.load_state_dict({key[len(name) + 1:]: value for key, value in tensors.items()
                                  if key.startswith(name + ".")})
        with torch.no_grad():
            model.logit_scale.copy_(tensors["logit_scale"])
        return model

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for name, head in (("state_head", self.state_head), ("action_head", self.action_head)):
            tensors.update({f"{name}.{key}": value.detach().cpu().contiguous()
                            for key, value in head.state_dict().items()})
        tensors["logit_scale"] = self.logit_scale.detach().cpu().reshape(()).clone()
        save_file(tensors, directory / HEAD_FILE)
        (directory / HEAD_CONFIG_FILE).write_text(json.dumps(self.arch, indent=2), encoding="utf-8")

    # ── 계산 ──
    def parameters(self) -> list:
        return list(self.state_head.parameters()) + list(self.action_head.parameters()) + [self.logit_scale]

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def train(self) -> None:
        self.state_head.train()
        self.action_head.train()

    def eval(self) -> None:
        self.state_head.eval()
        self.action_head.eval()

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def logits(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """state [B, 4096], candidates [B, K, 4096] -> 선택지별 점수 [B, K]"""
        state_z = F.normalize(self.state_head(state), dim=-1)
        batch, choices = candidates.shape[0], candidates.shape[1]
        candidate_z = F.normalize(self.action_head(candidates.reshape(-1, candidates.shape[-1])), dim=-1)
        return self.scale() * torch.einsum("bh,bkh->bk", state_z, candidate_z.view(batch, choices, -1))

    @torch.no_grad()
    def score(self, state: np.ndarray, candidates: np.ndarray) -> tuple[torch.Tensor, float]:
        """넘파이 임베딩 -> (선택지별 점수 [B, K], 계산 시간(초)). candidates는 [K, 4096](모든 문장 공통) 또는 [B, K, 4096]."""
        self.eval()
        started = time.perf_counter()
        state_t = torch.from_numpy(state).to(self.device)
        candidate_t = torch.from_numpy(candidates).to(self.device)
        if candidate_t.dim() == 2:
            candidate_t = candidate_t.unsqueeze(0).expand(state_t.shape[0], -1, -1)
        logits = self.logits(state_t, candidate_t).cpu()
        return logits, time.perf_counter() - started


def infonce_loss(model: HeadModel, emb: torch.Tensor, state_idx: torch.Tensor, option_idx: torch.Tensor,
                 option_target: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """같은 배치 안의 서로 다른 선택지 텍스트를 negative로 쓰는 양방향 InfoNCE.

    문장 -> 선택지는 정답 분포를, 선택지 -> 문장은 그 선택지를 정답으로 가진 문장들을 목표로 한다. (CLM run_choice와 같은 식)
    emb: 학습 데이터의 서로 다른 텍스트 임베딩 [T, 4096] / state_idx: 문장별 emb 행 번호 [N]
    option_idx: 문장별 선택지의 emb 행 번호 [N, Kmax] (없는 칸은 -1) / option_target: 정답 분포 [N, Kmax]
    """
    options, target = option_idx[idx], option_target[idx]
    valid = options >= 0
    pool, column = torch.unique(options[valid], return_inverse=True)
    columns = torch.zeros_like(options)
    columns[valid] = column
    matrix = torch.zeros(len(idx), len(pool), device=emb.device).scatter_add_(1, columns, target * valid)
    state_z = F.normalize(model.state_head(emb[state_idx[idx]]), dim=-1)
    option_z = F.normalize(model.action_head(emb[pool]), dim=-1)
    logits = model.scale() * state_z @ option_z.t()
    forward = -(matrix * F.log_softmax(logits, 1)).sum(1).mean()
    weight = matrix.t()
    keep = weight.sum(1) > 0
    weight = weight[keep] / weight[keep].sum(1, keepdim=True)
    backward = -(weight * F.log_softmax(logits.t()[keep], 1)).sum(1).mean()
    return (forward + backward) / 2


def check_same_encoder(saved: dict, current: dict) -> None:
    """헤드를 학습할 때 쓴 인코더(encoder.json)와 지금 인코더가 같은지 확인한다. 다르면 점수가 의미 없다."""
    problems = []
    for key in ("gguf_file", "pooling", "max_len", "hidden_size", "tokenizer"):
        if saved.get(key) != current.get(key):
            problems.append(f"{key}: 학습 {saved.get(key)!r} / 지금 {current.get(key)!r}")
    if saved.get("gguf_size_bytes") and current.get("gguf_size_bytes") \
            and saved["gguf_size_bytes"] != current["gguf_size_bytes"]:
        problems.append(f"gguf_size_bytes: 학습 {saved['gguf_size_bytes']} / 지금 {current['gguf_size_bytes']}")
    if problems:
        raise ValueError("이 헤드는 지금 인코더와 짝이 맞지 않습니다.\n  " + "\n  ".join(problems))


class ClmClassifier:
    """헤드 + 인코더로 문장을 분류한다. 질문(instructions)과 선택지 설명은 학습 때와 같은 것을 써야 한다."""

    def __init__(self, head: HeadModel, questions: dict, encoder, qid: str = "label"):
        self.head, self.questions, self.encoder, self.qid = head, questions, encoder, qid
        keys, candidates = load_clm().schema.build_pairs("", questions)[qid][1:]
        self.keys, self.candidate_texts = keys, candidates

    @classmethod
    def from_best_dir(cls, best_dir: Path, encoder, device="cpu") -> "ClmClassifier":
        best_dir = Path(best_dir)
        check_same_encoder(json.loads((best_dir / ENCODER_FILE).read_text(encoding="utf-8")), encoder.describe())
        questions = json.loads((best_dir / QUESTIONS_FILE).read_text(encoding="utf-8"))
        return cls(HeadModel.load(best_dir, device), questions, encoder)

    @classmethod
    def from_base_head(cls, questions: dict, encoder, device="cpu") -> "ClmClassifier":
        """파인튜닝하지 않은 공식 헤드 (zero-shot)."""
        return cls(HeadModel.from_checkpoint(base_head_path(), device), questions, encoder)

    def state_texts(self, texts: list[str]) -> list[str]:
        build_pairs = load_clm().schema.build_pairs
        return [build_pairs(text, self.questions)[self.qid][0] for text in texts]

    def predict_logits(self, texts: list[str]) -> tuple[torch.Tensor, float]:
        """-> (선택지별 점수 [N, K], 걸린 시간(초))
        시간 = 문장별 인코더 계산 시간의 합 + 헤드 계산 시간. 선택지 설명은 한 번만 계산해 두면 되므로 넣지 않는다."""
        state_vectors, encode_seconds = self.encoder.embed(self.state_texts(texts))
        candidate_vectors, _ = self.encoder.embed(self.candidate_texts)
        logits, head_seconds = self.head.score(state_vectors, candidate_vectors)
        return logits, float(encode_seconds.sum()) + head_seconds

    def predict(self, texts: list[str]) -> tuple[list[int], list[list[float]], float]:
        """-> (예측한 선택지 번호, 선택지별 확률, 걸린 시간(초))"""
        logits, seconds = self.predict_logits(texts)
        return logits.argmax(-1).tolist(), torch.softmax(logits, dim=-1).tolist(), seconds
