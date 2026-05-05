from __future__ import annotations

import json
import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32
DEVICE = torch.device("cpu")

def clamp(x, a, b):
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, a, b)
    return max(a, min(b, x))

def clamp01(x):
    return clamp(x, 0.0, 1.0)

def clamp11(x):
    return clamp(x, -1.0, 1.0)

def to_tensor(x, device=DEVICE, dtype=DTYPE):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def safe_norm(x, dim=-1, keepdim=False, eps=1e-8):
    if isinstance(x, torch.Tensor):
        return torch.norm(x, p=2, dim=dim, keepdim=keepdim).clamp_min(eps)
    arr = np.asarray(x, dtype=np.float32)
    return float(max(np.linalg.norm(arr), eps))

def lerp(a, b, t):
    return a + (b - a) * t

def exp_smooth(current, target, dt, tau):
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-float(dt) / float(tau))
    return lerp(current, target, k)

def normalize(v, dim=-1, eps=1e-8):
    if isinstance(v, torch.Tensor):
        return v / safe_norm(v, dim=dim, keepdim=True, eps=eps)
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=dim, keepdims=True) + eps)

def cosine_similarity(a, b, eps=1e-8):
    a = to_tensor(a); b = to_tensor(b)
    return torch.sum(a * b, dim=-1) / (safe_norm(a, dim=-1) * safe_norm(b, dim=-1) + eps)

def identity(n):
    return torch.eye(n, dtype=DTYPE, device=DEVICE)

def rot2(theta):
    t = torch.as_tensor(theta, dtype=DTYPE, device=DEVICE)
    c = torch.cos(t); s = torch.sin(t)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

def affine2(sx=1.0, sy=1.0, shx=0.0, shy=0.0, theta=0.0, tx=0.0, ty=0.0):
    S = torch.tensor([[sx, shx], [shy, sy]], dtype=DTYPE, device=DEVICE)
    R = rot2(theta)
    A = R @ S
    b = torch.tensor([tx, ty], dtype=DTYPE, device=DEVICE)
    return A, b

def apply_affine(points, A, b):
    points = to_tensor(points)
    return points @ A.T + b

def attention_mask_causal(seq_len: int) -> torch.Tensor:
    m = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE, dtype=DTYPE), diagonal=1)
    return m * -1e9

def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim, dtype=DTYPE, device=DEVICE)
    position = torch.arange(0, length, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=DTYPE, device=DEVICE) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Kind(str, Enum):
    motion = "motion"
    gaze = "gaze"
    face = "face"
    memory = "memory"
    vision = "vision"
    language = "language"
    system = "system"
    pedagogy = "pedagogy"
    code = "code"
    audio = "audio"
    trend = "trend"
    privacy = "privacy"
    fine_tuning = "fine_tuning"

def vec2(x: float, y: float) -> torch.Tensor:
    return torch.tensor([x, y], dtype=DTYPE, device=DEVICE)

def vec3(x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor([x, y, z], dtype=DTYPE, device=DEVICE)

def rot3_x(theta):
    t = torch.as_tensor(theta, dtype=DTYPE, device=DEVICE)
    c = torch.cos(t); s = torch.sin(t)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=DTYPE, device=DEVICE)

def rot3_y(theta):
    t = torch.as_tensor(theta, dtype=DTYPE, device=DEVICE)
    c = torch.cos(t); s = torch.sin(t)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=DTYPE, device=DEVICE)

def rot3_z(theta):
    t = torch.as_tensor(theta, dtype=DTYPE, device=DEVICE)
    c = torch.cos(t); s = torch.sin(t)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=DTYPE, device=DEVICE)

def spherical_to_cartesian(r, theta, phi):
    r = torch.as_tensor(r, device=DEVICE, dtype=DTYPE)
    theta = torch.as_tensor(theta, device=DEVICE, dtype=DTYPE)
    phi = torch.as_tensor(phi, device=DEVICE, dtype=DTYPE)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)

def cartesian_to_spherical(xyz: torch.Tensor, eps=1e-8):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x * x + y * y + z * z).clamp_min(eps)
    theta = torch.atan2(y, x)
    phi = torch.acos((z / r).clamp(-1.0, 1.0))
    return torch.stack([r, theta, phi], dim=-1)

def covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / max(1, x.shape[0] - 1)

def mahalanobis(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    diff = x - mean
    inv = torch.linalg.pinv(cov)
    return torch.sqrt((diff @ inv * diff).sum(-1).clamp_min(1e-8))

def gaussian_pdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    z = (x - mean) / std.clamp_min(1e-8)
    return torch.exp(-0.5 * z * z) / (std * math.sqrt(2 * math.pi))

def softplus01(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def exp_decay(value: torch.Tensor, rate: float, dt: float) -> torch.Tensor:
    return value * torch.exp(torch.tensor(-rate * dt, device=value.device, dtype=value.dtype))

def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim, dtype=DTYPE, device=DEVICE)
    position = torch.arange(0, length, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=DTYPE, device=DEVICE) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class LinearAlgebraKernel(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.matrix = nn.Parameter(torch.eye(dim, dtype=DTYPE))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=DTYPE))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.matrix.T + self.bias
        return self.norm(torch.tanh(y))
    def spectral_radius(self) -> torch.Tensor:
        return torch.linalg.norm(self.matrix, ord=2)
    def frobenius(self) -> torch.Tensor:
        return torch.linalg.norm(self.matrix, ord="fro")

def mat_001(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_001(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_001(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_001(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_001(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_001(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_001(x, basis)

def unit_001(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_002(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_002(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_002(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_002(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_002(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_002(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_002(x, basis)

def unit_002(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_003(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_003(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_003(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_003(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_003(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_003(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_003(x, basis)

def unit_003(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_004(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_004(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_004(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_004(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_004(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_004(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_004(x, basis)

def unit_004(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_005(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_005(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_005(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_005(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_005(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_005(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_005(x, basis)

def unit_005(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_006(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_006(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_006(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_006(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_006(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_006(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_006(x, basis)

def unit_006(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_007(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_007(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_007(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_007(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_007(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_007(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_007(x, basis)

def unit_007(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_008(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_008(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_008(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_008(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_008(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_008(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_008(x, basis)

def unit_008(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_009(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_009(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_009(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_009(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_009(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_009(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_009(x, basis)

def unit_009(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_010(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_010(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_010(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_010(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_010(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_010(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_010(x, basis)

def unit_010(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_011(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_011(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_011(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_011(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_011(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_011(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_011(x, basis)

def unit_011(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_012(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_012(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_012(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_012(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_012(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_012(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_012(x, basis)

def unit_012(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_013(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_013(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_013(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_013(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_013(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_013(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_013(x, basis)

def unit_013(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_014(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_014(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_014(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_014(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_014(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_014(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_014(x, basis)

def unit_014(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_015(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_015(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_015(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_015(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_015(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_015(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_015(x, basis)

def unit_015(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_016(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_016(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_016(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_016(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_016(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_016(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_016(x, basis)

def unit_016(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_017(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_017(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_017(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_017(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_017(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_017(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_017(x, basis)

def unit_017(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_018(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_018(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_018(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_018(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_018(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_018(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_018(x, basis)

def unit_018(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_019(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_019(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_019(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_019(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_019(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_019(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_019(x, basis)

def unit_019(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_020(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_020(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_020(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_020(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_020(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_020(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_020(x, basis)

def unit_020(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_021(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_021(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_021(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_021(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_021(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_021(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_021(x, basis)

def unit_021(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_022(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_022(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_022(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_022(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_022(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_022(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_022(x, basis)

def unit_022(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_023(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_023(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_023(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_023(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_023(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_023(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_023(x, basis)

def unit_023(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_024(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_024(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_024(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_024(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_024(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_024(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_024(x, basis)

def unit_024(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_025(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_025(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_025(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_025(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_025(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_025(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_025(x, basis)

def unit_025(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_026(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_026(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_026(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_026(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_026(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_026(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_026(x, basis)

def unit_026(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_027(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_027(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_027(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_027(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_027(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_027(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_027(x, basis)

def unit_027(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_028(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_028(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_028(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_028(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_028(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_028(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_028(x, basis)

def unit_028(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_029(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_029(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_029(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_029(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_029(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_029(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_029(x, basis)

def unit_029(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_030(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_030(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_030(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_030(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_030(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_030(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_030(x, basis)

def unit_030(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_031(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_031(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_031(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_031(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_031(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_031(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_031(x, basis)

def unit_031(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_032(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_032(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_032(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_032(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_032(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_032(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_032(x, basis)

def unit_032(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_033(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_033(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_033(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_033(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_033(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_033(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_033(x, basis)

def unit_033(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_034(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_034(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_034(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_034(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_034(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_034(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_034(x, basis)

def unit_034(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_035(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_035(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_035(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_035(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_035(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_035(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_035(x, basis)

def unit_035(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_036(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_036(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_036(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_036(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_036(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_036(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_036(x, basis)

def unit_036(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_037(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_037(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_037(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_037(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_037(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_037(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_037(x, basis)

def unit_037(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_038(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_038(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_038(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_038(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_038(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_038(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_038(x, basis)

def unit_038(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_039(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_039(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_039(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_039(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_039(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_039(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_039(x, basis)

def unit_039(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_040(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_040(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_040(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_040(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_040(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_040(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_040(x, basis)

def unit_040(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_041(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_041(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_041(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_041(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_041(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_041(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_041(x, basis)

def unit_041(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_042(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_042(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_042(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_042(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_042(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_042(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_042(x, basis)

def unit_042(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_043(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_043(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_043(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_043(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_043(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_043(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_043(x, basis)

def unit_043(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_044(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_044(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_044(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_044(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_044(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_044(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_044(x, basis)

def unit_044(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_045(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_045(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_045(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_045(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_045(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_045(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_045(x, basis)

def unit_045(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_046(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_046(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_046(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_046(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_046(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_046(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_046(x, basis)

def unit_046(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_047(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_047(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_047(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_047(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_047(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_047(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_047(x, basis)

def unit_047(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_048(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_048(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_048(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_048(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_048(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_048(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_048(x, basis)

def unit_048(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_049(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_049(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_049(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_049(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_049(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_049(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_049(x, basis)

def unit_049(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_050(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_050(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_050(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_050(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_050(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_050(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_050(x, basis)

def unit_050(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_051(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_051(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_051(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_051(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_051(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_051(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_051(x, basis)

def unit_051(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_052(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_052(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_052(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_052(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_052(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_052(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_052(x, basis)

def unit_052(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_053(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_053(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_053(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_053(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_053(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_053(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_053(x, basis)

def unit_053(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_054(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_054(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_054(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_054(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_054(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_054(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_054(x, basis)

def unit_054(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_055(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_055(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_055(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_055(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_055(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_055(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_055(x, basis)

def unit_055(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_056(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_056(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_056(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_056(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_056(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_056(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_056(x, basis)

def unit_056(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_057(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_057(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_057(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_057(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_057(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_057(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_057(x, basis)

def unit_057(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_058(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_058(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_058(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_058(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_058(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_058(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_058(x, basis)

def unit_058(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_059(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_059(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_059(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_059(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_059(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_059(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_059(x, basis)

def unit_059(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_060(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_060(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_060(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_060(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_060(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_060(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_060(x, basis)

def unit_060(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_061(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_061(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_061(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_061(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_061(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_061(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_061(x, basis)

def unit_061(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_062(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_062(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_062(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_062(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_062(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_062(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_062(x, basis)

def unit_062(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_063(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_063(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_063(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_063(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_063(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_063(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_063(x, basis)

def unit_063(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_064(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_064(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_064(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_064(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_064(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_064(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_064(x, basis)

def unit_064(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_065(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_065(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_065(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_065(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_065(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_065(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_065(x, basis)

def unit_065(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_066(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_066(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_066(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_066(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_066(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_066(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_066(x, basis)

def unit_066(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_067(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_067(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_067(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_067(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_067(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_067(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_067(x, basis)

def unit_067(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_068(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_068(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_068(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_068(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_068(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_068(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_068(x, basis)

def unit_068(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_069(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_069(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_069(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_069(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_069(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_069(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_069(x, basis)

def unit_069(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_070(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_070(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_070(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_070(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_070(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_070(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_070(x, basis)

def unit_070(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_071(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_071(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_071(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_071(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_071(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_071(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_071(x, basis)

def unit_071(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_072(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_072(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_072(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_072(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_072(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_072(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_072(x, basis)

def unit_072(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_073(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_073(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_073(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_073(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_073(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_073(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_073(x, basis)

def unit_073(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_074(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_074(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_074(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_074(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_074(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_074(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_074(x, basis)

def unit_074(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_075(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_075(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_075(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_075(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_075(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_075(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_075(x, basis)

def unit_075(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_076(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_076(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_076(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_076(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_076(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_076(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_076(x, basis)

def unit_076(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_077(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_077(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_077(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_077(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_077(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_077(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_077(x, basis)

def unit_077(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_078(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_078(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_078(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_078(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_078(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_078(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_078(x, basis)

def unit_078(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_079(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_079(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_079(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_079(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_079(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_079(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_079(x, basis)

def unit_079(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_080(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_080(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_080(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_080(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_080(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_080(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_080(x, basis)

def unit_080(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_081(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_081(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_081(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_081(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_081(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_081(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_081(x, basis)

def unit_081(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_082(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_082(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_082(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_082(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_082(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_082(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_082(x, basis)

def unit_082(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_083(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_083(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_083(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_083(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_083(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_083(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_083(x, basis)

def unit_083(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_084(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_084(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_084(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_084(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_084(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_084(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_084(x, basis)

def unit_084(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_085(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_085(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_085(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_085(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_085(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_085(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_085(x, basis)

def unit_085(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_086(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_086(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_086(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_086(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_086(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_086(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_086(x, basis)

def unit_086(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_087(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_087(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_087(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_087(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_087(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_087(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_087(x, basis)

def unit_087(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_088(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_088(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_088(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_088(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_088(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_088(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_088(x, basis)

def unit_088(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_089(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_089(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_089(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_089(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_089(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_089(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_089(x, basis)

def unit_089(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_090(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_090(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_090(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_090(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_090(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_090(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_090(x, basis)

def unit_090(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_091(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_091(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_091(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_091(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_091(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_091(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_091(x, basis)

def unit_091(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_092(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_092(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_092(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_092(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_092(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_092(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_092(x, basis)

def unit_092(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_093(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_093(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_093(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_093(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_093(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_093(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_093(x, basis)

def unit_093(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_094(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_094(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_094(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_094(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_094(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_094(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_094(x, basis)

def unit_094(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_095(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_095(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_095(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_095(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_095(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_095(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_095(x, basis)

def unit_095(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_096(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_096(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_096(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_096(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_096(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_096(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_096(x, basis)

def unit_096(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_097(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_097(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_097(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_097(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_097(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_097(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_097(x, basis)

def unit_097(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_098(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_098(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_098(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_098(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_098(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_098(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_098(x, basis)

def unit_098(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_099(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_099(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_099(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_099(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_099(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_099(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_099(x, basis)

def unit_099(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_100(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_100(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_100(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_100(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_100(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_100(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_100(x, basis)

def unit_100(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_101(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_101(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_101(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_101(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_101(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_101(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_101(x, basis)

def unit_101(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_102(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_102(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_102(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_102(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_102(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_102(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_102(x, basis)

def unit_102(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_103(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_103(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_103(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_103(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_103(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_103(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_103(x, basis)

def unit_103(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_104(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_104(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_104(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_104(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_104(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_104(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_104(x, basis)

def unit_104(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_105(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_105(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_105(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_105(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_105(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_105(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_105(x, basis)

def unit_105(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_106(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_106(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_106(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_106(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_106(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_106(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_106(x, basis)

def unit_106(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_107(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_107(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_107(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_107(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_107(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_107(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_107(x, basis)

def unit_107(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_108(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_108(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_108(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_108(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_108(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_108(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_108(x, basis)

def unit_108(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_109(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_109(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_109(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_109(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_109(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_109(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_109(x, basis)

def unit_109(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_110(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_110(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_110(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_110(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_110(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_110(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_110(x, basis)

def unit_110(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_111(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_111(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_111(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_111(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_111(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_111(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_111(x, basis)

def unit_111(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_112(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_112(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_112(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_112(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_112(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_112(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_112(x, basis)

def unit_112(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_113(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_113(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_113(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_113(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_113(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_113(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_113(x, basis)

def unit_113(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_114(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_114(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_114(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_114(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_114(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_114(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_114(x, basis)

def unit_114(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_115(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_115(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_115(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_115(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_115(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_115(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_115(x, basis)

def unit_115(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_116(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_116(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_116(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_116(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_116(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_116(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_116(x, basis)

def unit_116(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_117(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_117(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_117(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_117(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_117(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_117(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_117(x, basis)

def unit_117(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_118(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_118(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_118(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_118(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_118(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_118(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_118(x, basis)

def unit_118(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_119(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_119(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_119(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_119(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_119(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_119(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_119(x, basis)

def unit_119(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_120(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_120(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_120(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_120(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_120(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_120(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_120(x, basis)

def unit_120(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_121(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_121(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_121(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_121(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_121(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_121(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_121(x, basis)

def unit_121(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_122(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_122(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_122(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_122(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_122(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_122(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_122(x, basis)

def unit_122(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_123(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_123(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_123(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_123(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_123(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_123(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_123(x, basis)

def unit_123(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_124(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_124(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_124(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_124(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_124(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_124(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_124(x, basis)

def unit_124(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_125(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_125(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_125(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_125(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_125(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_125(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_125(x, basis)

def unit_125(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_126(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_126(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_126(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_126(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_126(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_126(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_126(x, basis)

def unit_126(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_127(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_127(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_127(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_127(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_127(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_127(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_127(x, basis)

def unit_127(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_128(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_128(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_128(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_128(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_128(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_128(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_128(x, basis)

def unit_128(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_129(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_129(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_129(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_129(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_129(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_129(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_129(x, basis)

def unit_129(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_130(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_130(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_130(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_130(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_130(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_130(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_130(x, basis)

def unit_130(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_131(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_131(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_131(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_131(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_131(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_131(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_131(x, basis)

def unit_131(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_132(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_132(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_132(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_132(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_132(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_132(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_132(x, basis)

def unit_132(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_133(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_133(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_133(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_133(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_133(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_133(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_133(x, basis)

def unit_133(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_134(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_134(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_134(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_134(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_134(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_134(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_134(x, basis)

def unit_134(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_135(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_135(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_135(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_135(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_135(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_135(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_135(x, basis)

def unit_135(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_136(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_136(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_136(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_136(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_136(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_136(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_136(x, basis)

def unit_136(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_137(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_137(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_137(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_137(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_137(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_137(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_137(x, basis)

def unit_137(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_138(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_138(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_138(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_138(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_138(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_138(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_138(x, basis)

def unit_138(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_139(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_139(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_139(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_139(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_139(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_139(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_139(x, basis)

def unit_139(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_140(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_140(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_140(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_140(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_140(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_140(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_140(x, basis)

def unit_140(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_141(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_141(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_141(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_141(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_141(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_141(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_141(x, basis)

def unit_141(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_142(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_142(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_142(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_142(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_142(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_142(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_142(x, basis)

def unit_142(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_143(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_143(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_143(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_143(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_143(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_143(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_143(x, basis)

def unit_143(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_144(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_144(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_144(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_144(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_144(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_144(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_144(x, basis)

def unit_144(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_145(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_145(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_145(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_145(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_145(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_145(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_145(x, basis)

def unit_145(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_146(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_146(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_146(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_146(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_146(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_146(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_146(x, basis)

def unit_146(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_147(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_147(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_147(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_147(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_147(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_147(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_147(x, basis)

def unit_147(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_148(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_148(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_148(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_148(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_148(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_148(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_148(x, basis)

def unit_148(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_149(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_149(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_149(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_149(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_149(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_149(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_149(x, basis)

def unit_149(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_150(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_150(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_150(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_150(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_150(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_150(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_150(x, basis)

def unit_150(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_151(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_151(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_151(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_151(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_151(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_151(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_151(x, basis)

def unit_151(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_152(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_152(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_152(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_152(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_152(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_152(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_152(x, basis)

def unit_152(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_153(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_153(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_153(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_153(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_153(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_153(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_153(x, basis)

def unit_153(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_154(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_154(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_154(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_154(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_154(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_154(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_154(x, basis)

def unit_154(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_155(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_155(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_155(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_155(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_155(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_155(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_155(x, basis)

def unit_155(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_156(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_156(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_156(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_156(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_156(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_156(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_156(x, basis)

def unit_156(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_157(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_157(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_157(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_157(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_157(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_157(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_157(x, basis)

def unit_157(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_158(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_158(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_158(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_158(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_158(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_158(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_158(x, basis)

def unit_158(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_159(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_159(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_159(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_159(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_159(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_159(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_159(x, basis)

def unit_159(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_160(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_160(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_160(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_160(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_160(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_160(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_160(x, basis)

def unit_160(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_161(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_161(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_161(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_161(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_161(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_161(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_161(x, basis)

def unit_161(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_162(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_162(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_162(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_162(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_162(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_162(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_162(x, basis)

def unit_162(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_163(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_163(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_163(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_163(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_163(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_163(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_163(x, basis)

def unit_163(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_164(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_164(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_164(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_164(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_164(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_164(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_164(x, basis)

def unit_164(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_165(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_165(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_165(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_165(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_165(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_165(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_165(x, basis)

def unit_165(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_166(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_166(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_166(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_166(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_166(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_166(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_166(x, basis)

def unit_166(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_167(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_167(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_167(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_167(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_167(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_167(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_167(x, basis)

def unit_167(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_168(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_168(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_168(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_168(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_168(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_168(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_168(x, basis)

def unit_168(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_169(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_169(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_169(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_169(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_169(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_169(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_169(x, basis)

def unit_169(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_170(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_170(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_170(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_170(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_170(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_170(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_170(x, basis)

def unit_170(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_171(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_171(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_171(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_171(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_171(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_171(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_171(x, basis)

def unit_171(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_172(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_172(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_172(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_172(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_172(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_172(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_172(x, basis)

def unit_172(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_173(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_173(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_173(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_173(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_173(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_173(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_173(x, basis)

def unit_173(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_174(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_174(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_174(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.60)

def blend_174(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_174(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_174(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_174(x, basis)

def unit_174(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_175(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_175(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_175(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.00)

def blend_175(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_175(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_175(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_175(x, basis)

def unit_175(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_176(value: float = 1.0) -> torch.Tensor:
    return torch.eye(3, dtype=DTYPE, device=DEVICE) * value

def vec_176(value: float = 0.0) -> torch.Tensor:
    return torch.full((3,), value, dtype=DTYPE, device=DEVICE)

def op_176(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.10)

def blend_176(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_176(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_176(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_176(x, basis)

def unit_176(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_177(value: float = 1.0) -> torch.Tensor:
    return torch.eye(4, dtype=DTYPE, device=DEVICE) * value

def vec_177(value: float = 0.0) -> torch.Tensor:
    return torch.full((4,), value, dtype=DTYPE, device=DEVICE)

def op_177(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.20)

def blend_177(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_177(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_177(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_177(x, basis)

def unit_177(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_178(value: float = 1.0) -> torch.Tensor:
    return torch.eye(5, dtype=DTYPE, device=DEVICE) * value

def vec_178(value: float = 0.0) -> torch.Tensor:
    return torch.full((5,), value, dtype=DTYPE, device=DEVICE)

def op_178(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.30)

def blend_178(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_178(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_178(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_178(x, basis)

def unit_178(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_179(value: float = 1.0) -> torch.Tensor:
    return torch.eye(6, dtype=DTYPE, device=DEVICE) * value

def vec_179(value: float = 0.0) -> torch.Tensor:
    return torch.full((6,), value, dtype=DTYPE, device=DEVICE)

def op_179(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.40)

def blend_179(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_179(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_179(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_179(x, basis)

def unit_179(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

def mat_180(value: float = 1.0) -> torch.Tensor:
    return torch.eye(2, dtype=DTYPE, device=DEVICE) * value

def vec_180(value: float = 0.0) -> torch.Tensor:
    return torch.full((2,), value, dtype=DTYPE, device=DEVICE)

def op_180(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.tanh((x + y) * 1.50)

def blend_180(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    return lerp(a, b, t)

def project_180(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeff = torch.matmul(x, basis)
    return torch.matmul(coeff, basis.T)

def residual_180(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_180(x, basis)

def unit_180(x: torch.Tensor) -> torch.Tensor:
    return normalize(x)

# ======================================================================
# Commercial math layer for Denise vNext
# ======================================================================

def fourier_schedule_179(t: torch.Tensor, *, freqs: Optional[Sequence[float]] = None, amps: Optional[Sequence[float]] = None, phases: Optional[Sequence[float]] = None, base: float = 0.5) -> torch.Tensor:
    t = torch.as_tensor(t, dtype=DTYPE, device=DEVICE)
    freqs = list(freqs) if freqs is not None else [0.37, 0.73, 1.11, 1.91]
    amps = list(amps) if amps is not None else [0.33, 0.19, 0.11, 0.07]
    phases = list(phases) if phases is not None else [0.0, 0.7, 1.4, 2.2]
    out = torch.full_like(t, float(base))
    for f, a, p in zip(freqs, amps, phases):
        out = out + float(a) * torch.sin(t * float(f) + float(p))
    return out


def spring_step_179(position: torch.Tensor, velocity: torch.Tensor, target: torch.Tensor, dt: float, stiffness: float = 12.0, damping: float = 0.82) -> Tuple[torch.Tensor, torch.Tensor]:
    position = to_tensor(position)
    velocity = to_tensor(velocity)
    target = to_tensor(target)
    force = (target - position) * float(stiffness)
    velocity = (velocity + force * float(dt)) * float(damping)
    position = position + velocity * float(dt)
    return position, velocity


def spline_179(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(t, dtype=DTYPE, device=DEVICE)
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - c) * t2 + (-a + 3 * b - 3 * c + c) * t3)


def project_weighted_179(x: torch.Tensor, basis: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    x = to_tensor(x)
    basis = to_tensor(basis)
    weights = to_tensor(weights)
    coeff = torch.matmul(x, basis) * weights
    return torch.matmul(coeff, basis.T)


def tensor_slot_score_179(free_space: torch.Tensor, overlap_penalty: torch.Tensor, mouse_velocity_field: torch.Tensor, attention_reward: torch.Tensor) -> torch.Tensor:
    return op_179(op_179(free_space, -overlap_penalty), op_179(-mouse_velocity_field, attention_reward))


__all__.extend([
    'fourier_schedule_179',
    'spring_step_179',
    'spline_179',
    'project_weighted_179',
    'tensor_slot_score_179',
])
