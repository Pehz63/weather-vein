from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOCAL_WFCRL = REPO_ROOT / "wfcrl-env"
if LOCAL_WFCRL.exists():
    sys.path.insert(0, str(LOCAL_WFCRL))

# Keep WFCRL-generated simulation folders under CSCI_5980_notebooks, matching the notebooks.
os.chdir(SCRIPT_DIR)

from wfcrl import environments as envs  # noqa: E402


FIXED_FLORIS_TEST_OPTIONS = {"wind_speed": 8.0, "wind_direction": 270.0}
SCENARIO2_WEIBULL_SHAPE = 8.0
SCENARIO2_WEIBULL_SCALE = 8.0
SCENARIO2_DIRECTION_MEAN = 270.0
SCENARIO2_DIRECTION_STD = 20.0
GRAPH_YAW_SCALE = 30.0
GRAPH_WS_SCALE = 15.0

SLIDE_LAYOUTS = [
    "Ablaincourt_Floris",
    "HornsRev1_Floris",
    "HornsRev2_Floris",
    "Ormonde_Floris",
    "Turb10_Row1_Floris",
    "Turb11_Row1_Floris",
    "Turb12_Row1_Floris",
    "Turb16_Row5_Floris",
    "Turb2_Row1_Floris",
    "Turb32_Row5_Floris",
    "Turb3_Row1_Floris",
    "Turb4_Row1_Floris",
    "Turb5_Row1_Floris",
    "Turb6_Row1_Floris",
    "Turb6_Row2_Floris",
    "Turb7_Row1_Floris",
    "Turb8_Row1_Floris",
    "Turb9_Row1_Floris",
    "Turb_TCRWP_Floris",
    "WMR_Floris",
]
FALLBACK_LAYOUTS = ["Turb3_Row1_Floris", "Ablaincourt_Floris"]


@dataclass
class GraphSample:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    y_total: float
    y_node: np.ndarray


@dataclass
class ScenarioConfig:
    name: str
    eval_options: dict
    context_sampler: Callable | None = None
    context_options: dict | None = None


def angle_wrap_deg(angle: float) -> float:
    return (angle + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def bearing_deg(dx: float, dy: float) -> float:
    # 0 deg = +y, 90 deg = +x.
    return angle_wrap_deg(math.degrees(math.atan2(dx, dy)))


def build_wake_graph(
    xy: np.ndarray,
    wind_dir_deg: float,
    dist_thresh: float = 1200.0,
    cone_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    src, dst, attrs = [], [], []
    downwind = angle_wrap_deg(wind_dir_deg + 180.0)

    for i, (xi, yi) in enumerate(xy):
        for j, (xj, yj) in enumerate(xy):
            if i == j:
                continue
            dx = float(xj - xi)
            dy = float(yj - yi)
            dist = math.hypot(dx, dy)
            if dist > dist_thresh:
                continue
            bearing = bearing_deg(dx, dy)
            offset = angle_diff_deg(bearing, downwind)
            if offset <= cone_deg:
                src.append(i)
                dst.append(j)
                attrs.append([dist, offset, dist / dist_thresh])

    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32) if attrs else np.zeros((0, 3), np.float32)
    return edge_index, edge_attr


def jensen_like_power(
    xy: np.ndarray,
    yaw_deg: np.ndarray,
    wind_speed: float,
    wind_dir_deg: float,
    rotor_diam: float = 126.0,
    ct: float = 0.8,
    k_wake: float = 0.05,
) -> np.ndarray:
    n = xy.shape[0]
    downwind = angle_wrap_deg(wind_dir_deg + 180.0)
    theta = math.radians(downwind)
    ux, uy = math.sin(theta), math.cos(theta)
    deficits2 = np.zeros(n, dtype=np.float64)
    yaw_rad = np.deg2rad(yaw_deg)
    yaw_gain = np.cos(yaw_rad) ** 1.88

    for i, (xi, yi) in enumerate(xy):
        for j, (xj, yj) in enumerate(xy):
            if i == j:
                continue
            dx, dy = xj - xi, yj - yi
            downwind_dist = dx * ux + dy * uy
            if downwind_dist <= 0:
                continue
            crosswind_x = dx - downwind_dist * ux
            crosswind_y = dy - downwind_dist * uy
            crosswind_dist = math.hypot(crosswind_x, crosswind_y)
            wake_radius = rotor_diam / 2.0 + k_wake * downwind_dist
            if crosswind_dist > wake_radius:
                continue
            yaw_factor = math.cos(yaw_rad[i]) ** 1.2
            denom = (1.0 + 2.0 * k_wake * downwind_dist / rotor_diam) ** 2
            deficit = (1.0 - math.sqrt(1.0 - ct * yaw_factor)) / denom
            deficits2[j] += deficit**2

    ws_eff = wind_speed * (1.0 - np.sqrt(deficits2))
    ws_eff = np.clip(ws_eff, 0.0, None)
    return (ws_eff**3 * yaw_gain).astype(np.float32)


def sample_synthetic_graph(n_turbines: int, rng: np.random.Generator) -> GraphSample:
    area = 2000.0
    layout = rng.choice(["random", "grid", "row"])
    if layout == "grid":
        side = int(math.ceil(math.sqrt(n_turbines)))
        xs = np.linspace(0, area, side)
        ys = np.linspace(0, area, side)
        xy = np.array([(x, y) for x in xs for y in ys], dtype=np.float32)[:n_turbines]
    elif layout == "row":
        xy = np.column_stack(
            [np.linspace(0, area, n_turbines), np.full(n_turbines, area * 0.5)]
        ).astype(np.float32)
    else:
        xy = rng.uniform(0, area, size=(n_turbines, 2)).astype(np.float32)

    wind_speed = float(rng.uniform(6.0, 12.0))
    wind_dir = float(rng.uniform(0.0, 360.0))
    yaw = rng.uniform(-25.0, 25.0, size=n_turbines).astype(np.float32)
    edge_index, edge_attr = build_wake_graph(xy, wind_dir)
    x = np.column_stack(
        [
            yaw,
            np.full(n_turbines, wind_speed, dtype=np.float32),
            np.full(n_turbines, wind_dir, dtype=np.float32),
            xy[:, 0],
            xy[:, 1],
        ]
    ).astype(np.float32)
    y_node = jensen_like_power(xy, yaw, wind_speed, wind_dir)
    return GraphSample(x=x, edge_index=edge_index, edge_attr=edge_attr, y_total=float(y_node.sum()), y_node=y_node)


def normalize_xy_pair(x_coord, y_coord) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_coord, dtype=np.float32).reshape(-1)
    y = np.asarray(y_coord, dtype=np.float32).reshape(-1)
    return (x - x.mean()) / (x.std() + 1e-6), (y - y.mean()) / (y.std() + 1e-6)


def encode_graph_node_features(yaws, wind_speed, wind_dir_deg, x_coord, y_coord) -> np.ndarray:
    yaws = np.asarray(yaws, dtype=np.float32).reshape(-1)
    x_norm, y_norm = normalize_xy_pair(x_coord, y_coord)
    wd_rad = np.deg2rad(float(wind_dir_deg))
    return np.column_stack(
        [
            yaws / GRAPH_YAW_SCALE,
            np.full(yaws.size, float(wind_speed) / GRAPH_WS_SCALE, dtype=np.float32),
            np.full(yaws.size, np.sin(wd_rad), dtype=np.float32),
            np.full(yaws.size, np.cos(wd_rad), dtype=np.float32),
            x_norm.astype(np.float32),
            y_norm.astype(np.float32),
        ]
    ).astype(np.float32)


def graph_sample_to_node_x(sample: GraphSample) -> np.ndarray:
    x = np.asarray(sample.x, dtype=np.float32)
    if x.shape[1] == 6:
        return x
    return encode_graph_node_features(x[:, 0], float(x[0, 1]), float(x[0, 2]), x[:, 3], x[:, 4])


def to_pyg(sample: GraphSample) -> Data:
    return Data(
        x=torch.tensor(graph_sample_to_node_x(sample), dtype=torch.float32),
        edge_index=torch.tensor(sample.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(sample.edge_attr, dtype=torch.float32),
    )


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hid: int = 64):
        super().__init__()
        self.g1 = GATv2Conv(in_dim, hid, heads=2, concat=False, edge_dim=3)
        self.g2 = GATv2Conv(hid, hid, heads=2, concat=False, edge_dim=3)
        self.lin = nn.Linear(hid, hid)

    def forward(self, batch_data: Batch) -> torch.Tensor:
        h = self.g1(batch_data.x, batch_data.edge_index, batch_data.edge_attr).relu()
        h = self.g2(h, batch_data.edge_index, batch_data.edge_attr).relu()
        return self.lin(global_mean_pool(h, batch_data.batch))


class CrossAttentionRegressor(nn.Module):
    def __init__(self, emb: int = 64, heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(emb, emb)
        self.k_proj = nn.Linear(emb + 1, emb)
        self.v_proj = nn.Linear(emb + 1, emb)
        self.attn = nn.MultiheadAttention(embed_dim=emb, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, 1))

    def forward(self, q_emb, ctx_emb, ctx_y):
        q = self.q_proj(q_emb).unsqueeze(0)
        kv_in = torch.cat([ctx_emb, ctx_y], dim=-1).unsqueeze(0)
        out, _ = self.attn(q, self.k_proj(kv_in), self.v_proj(kv_in))
        return self.mlp(out.squeeze(0)).squeeze(-1)


class GraphPFNInContext(nn.Module):
    def __init__(self, node_dim: int = 6, emb: int = 64):
        super().__init__()
        self.enc = GraphEncoder(node_dim, hid=emb)
        self.head = CrossAttentionRegressor(emb=emb)

    def forward(self, ctx_batch: Batch, ctx_y: torch.Tensor, q_batch: Batch):
        return self.head(self.enc(q_batch), self.enc(ctx_batch), ctx_y)


def sample_graph_task(c_count: int, q_count: int, n_turbines: int, seed: int):
    rng = np.random.default_rng(seed)
    ctx = [sample_synthetic_graph(n_turbines, rng) for _ in range(c_count)]
    qry = [sample_synthetic_graph(n_turbines, rng) for _ in range(q_count)]
    ctx_y_raw = np.asarray([s.y_total for s in ctx], dtype=np.float32)
    q_y_raw = np.asarray([s.y_total for s in qry], dtype=np.float32)
    y_mean = float(ctx_y_raw.mean())
    y_std = float(ctx_y_raw.std() + 1e-6)
    return (
        Batch.from_data_list([to_pyg(s) for s in ctx]),
        torch.tensor(((ctx_y_raw - y_mean) / y_std).reshape(-1, 1), dtype=torch.float32),
        Batch.from_data_list([to_pyg(s) for s in qry]),
        torch.tensor((q_y_raw - y_mean) / y_std, dtype=torch.float32),
    )


def train_graphpfn(steps: int, device: torch.device, n_turbines: int = 9) -> GraphPFNInContext:
    model = GraphPFNInContext(node_dim=6, emb=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    for step in range(steps):
        ctx_batch, ctx_y, q_batch, q_y = sample_graph_task(16, 16, n_turbines=n_turbines, seed=step)
        ctx_batch = ctx_batch.to(device)
        q_batch = q_batch.to(device)
        ctx_y = ctx_y.to(device)
        q_y = q_y.to(device)
        pred = model(ctx_batch, ctx_y, q_batch)
        loss = loss_fn(pred, q_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"GraphPFN train step {step:03d}/{steps} loss={float(loss.item()):.4f}")
    return model


class SimpleTabPFNRegressor:
    """Hosted TabPFN if TABPFN_TOKEN works; ExtraTrees fallback otherwise."""

    def __init__(self, **kwargs):
        self.kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        self.display_name = "TabPFN"
        self.used_fallback = False
        self.model = None

    def fit(self, x, y):
        try:
            if not os.environ.get("TABPFN_TOKEN"):
                raise RuntimeError("TABPFN_TOKEN is not set")
            from tabpfn_client import TabPFNRegressor, set_access_token

            set_access_token(os.environ["TABPFN_TOKEN"])
            self.model = TabPFNRegressor(**self.kwargs)
            self.model.fit(x, y)
            self.display_name = "TabPFN"
            self.used_fallback = False
        except Exception as exc:
            print(f"Using TabPFN-Fallback instead of hosted TabPFN ({type(exc).__name__}).")
            self.model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=2, random_state=0, n_jobs=-1)
            self.model.fit(x, y)
            self.display_name = "TabPFN-Fallback"
            self.used_fallback = True
        return self

    def predict(self, x):
        return self.model.predict(x)


def load_tabpfn_token_from_colab_secret() -> None:
    try:
        from google.colab import userdata

        token = userdata.get("TABPFN_TOKEN")
        if token:
            os.environ["TABPFN_TOKEN"] = token
    except Exception:
        pass


def make_env(env_name: str, max_steps: int):
    return envs.make(
        env_name,
        max_num_steps=max_steps,
        controls={"yaw": (-5, 5)},
        continuous_control=True,
        log=True,
    )


def to_plain_obs(reset_or_step_out):
    if isinstance(reset_or_step_out, tuple):
        return reset_or_step_out[0]
    return reset_or_step_out


def scalarize_reward(reward) -> float:
    arr = np.asarray(reward, dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def safe_getattr(obj, name, default=None):
    if obj is None:
        return default
    try:
        getter = object.__getattribute__(obj, "get_wrapper_attr")
        value = getter(name)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        return object.__getattribute__(obj, name)
    except Exception:
        return default


def candidate_objects(env):
    objects, seen = [], set()

    def add(obj):
        if obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            objects.append(obj)

    add(env)
    add(safe_getattr(env, "unwrapped", None))
    idx = 0
    while idx < len(objects):
        obj = objects[idx]
        idx += 1
        for name in ("farm", "case", "farm_case", "config"):
            add(safe_getattr(obj, name, None))
    return objects


def get_num_turbines(env, obs=None) -> int:
    if isinstance(obs, dict) and "yaw" in obs:
        return int(np.asarray(obs["yaw"]).reshape(-1).size)
    for obj in candidate_objects(env):
        for attr in ("num_turbines", "n_turbines", "n_turbs"):
            value = safe_getattr(obj, attr, None)
            if value is not None:
                return int(value)
    raise AttributeError("Could not infer turbine count.")


def valid_layout_pair(x, y, n_expected: int):
    try:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if x.size == y.size == n_expected:
        return x, y
    return None


def get_layout_xy(env, obs=None):
    n_expected = get_num_turbines(env, obs)
    pairs = [
        ("layout_x", "layout_y"),
        ("x", "y"),
        ("turbine_x", "turbine_y"),
        ("coordinates_x", "coordinates_y"),
        ("xcoords", "ycoords"),
    ]
    if isinstance(obs, dict):
        for x_key, y_key in pairs:
            if x_key in obs and y_key in obs:
                pair = valid_layout_pair(obs[x_key], obs[y_key], n_expected)
                if pair is not None:
                    return pair
    for obj in candidate_objects(env):
        for x_attr, y_attr in pairs:
            pair = valid_layout_pair(safe_getattr(obj, x_attr), safe_getattr(obj, y_attr), n_expected)
            if pair is not None:
                return pair
    print(f"Warning: falling back to synthetic row coordinates for {n_expected} turbines.")
    return np.arange(n_expected, dtype=np.float32) * 500.0, np.zeros(n_expected, dtype=np.float32)


def extract_freewind(obs) -> tuple[float, float]:
    if "freewind_measurements" in obs:
        freewind = np.asarray(obs["freewind_measurements"], dtype=np.float32).reshape(-1)
        if freewind.size >= 2:
            return float(freewind[0]), float(freewind[1])
    return float(np.asarray(obs.get("wind_speed", 8.0)).reshape(-1)[0]), float(
        np.asarray(obs.get("wind_direction", 270.0)).reshape(-1)[0]
    )


def reset_env(env, seed: int, options: dict | None = None):
    return to_plain_obs(env.reset(seed=seed, options=dict(options) if options else None))


def freewind_bounds(env=None):
    low = np.array([3.0, 0.0], dtype=np.float32)
    high = np.array([28.0, 360.0], dtype=np.float32)
    try:
        space = env.observation_space["freewind_measurements"]
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)[:2]
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)[:2]
    except Exception:
        pass
    return low, high


def sample_scenario2_options(rng: np.random.Generator, env=None) -> dict:
    low, high = freewind_bounds(env)
    wind_speed = SCENARIO2_WEIBULL_SCALE * rng.weibull(SCENARIO2_WEIBULL_SHAPE)
    wind_direction = rng.normal(SCENARIO2_DIRECTION_MEAN, SCENARIO2_DIRECTION_STD) % 360.0
    return {
        "wind_speed": float(np.clip(wind_speed, low[0], high[0])),
        "wind_direction": float(np.clip(wind_direction, low[1], high[1])),
    }


def ramp_and_eval(env, target_yaws, seed: int, options: dict, max_steps: int) -> tuple[float, float]:
    obs = reset_env(env, seed=seed, options=options)
    total_reward = 0.0
    total_power = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps + 5:
        curr = np.asarray(obs["yaw"], dtype=np.float32)
        delta = np.clip(np.asarray(target_yaws, dtype=np.float32) - curr, -5.0, 5.0)
        obs, reward, term, trunc, info = env.step({"yaw": delta})
        obs = to_plain_obs(obs)
        total_reward += scalarize_reward(reward)
        if isinstance(info, dict) and "power" in info:
            total_power += float(np.sum(info["power"]))
        done = bool(term or trunc)
        steps += 1
    return total_reward, total_power


def wfcrl_graph_from_yaws(env, yaws, wind_speed, wind_dir, obs=None) -> GraphSample:
    x_coord, y_coord = get_layout_xy(env, obs)
    xy = np.column_stack([x_coord, y_coord]).astype(np.float32)
    edge_index, edge_attr = build_wake_graph(xy, wind_dir_deg=float(wind_dir))
    node_x = encode_graph_node_features(yaws, wind_speed, wind_dir, x_coord, y_coord)
    return GraphSample(
        x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_total=np.nan,
        y_node=np.zeros(node_x.shape[0], dtype=np.float32),
    )


def flatten_tabular_features(yaws, env, wind_speed, wind_dir, obs=None) -> np.ndarray:
    x_coord, y_coord = get_layout_xy(env, obs)
    yaws = np.asarray(yaws, dtype=np.float32).reshape(-1)
    wd_rad = np.deg2rad(float(wind_dir))
    x_norm, y_norm = normalize_xy_pair(x_coord, y_coord)
    return np.concatenate(
        [
            (yaws / GRAPH_YAW_SCALE).astype(np.float32),
            np.array([float(wind_speed) / GRAPH_WS_SCALE, np.sin(wd_rad), np.cos(wd_rad)], dtype=np.float32),
            x_norm.astype(np.float32),
            y_norm.astype(np.float32),
        ]
    ).astype(np.float32)


def collect_context_data(env, seed: int, n_initial: int, scenario: ScenarioConfig, max_steps: int):
    obs0 = reset_env(env, seed=seed, options=scenario.eval_options)
    ws, wd = extract_freewind(obs0)
    n_turbines = get_num_turbines(env, obs0)
    rng = np.random.default_rng(seed)

    x_tab, y_reward, y_power, graphs, yaws_list = [], [], [], [], []
    context_ws, context_wd = [], []

    for sample_idx in range(n_initial):
        yaws = rng.uniform(-40.0, 40.0, size=n_turbines).astype(np.float32)
        if scenario.context_sampler:
            options = scenario.context_sampler(rng, env)
        else:
            options = scenario.context_options or scenario.eval_options
        sample_seed = seed + sample_idx + 1
        obs_i = reset_env(env, seed=sample_seed, options=options)
        ws_i, wd_i = extract_freewind(obs_i)
        reward, power = ramp_and_eval(env, yaws, seed=sample_seed, options=options, max_steps=max_steps)

        x_tab.append(flatten_tabular_features(yaws, env, ws_i, wd_i, obs=obs_i))
        graphs.append(wfcrl_graph_from_yaws(env, yaws, ws_i, wd_i, obs=obs_i))
        y_reward.append(reward)
        y_power.append(power)
        yaws_list.append(yaws)
        context_ws.append(ws_i)
        context_wd.append(wd_i)

    return {
        "obs0": obs0,
        "ws": ws,
        "wd": wd,
        "x_tab": np.asarray(x_tab, dtype=np.float32),
        "y_reward": np.asarray(y_reward, dtype=np.float32),
        "y_power": np.asarray(y_power, dtype=np.float32),
        "graphs": graphs,
        "yaws": yaws_list,
        "context_ws": np.asarray(context_ws, dtype=np.float32),
        "context_wd": np.asarray(context_wd, dtype=np.float32),
    }


def fit_gp_model(x, y):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gp", GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)),
        ]
    )
    model.fit(x, y)
    return model


def score_graphpfn_candidates(
    graph_pfn_model,
    ctx_graphs,
    ctx_y,
    candidate_graphs,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    graph_pfn_model.eval()
    ctx_y_np = np.asarray(ctx_y, dtype=np.float32).reshape(-1)
    y_mean = float(ctx_y_np.mean())
    y_std = float(ctx_y_np.std() + 1e-6)
    ctx_y_norm = ((ctx_y_np - y_mean) / y_std).reshape(-1, 1)
    ctx_batch = Batch.from_data_list([to_pyg(g) for g in ctx_graphs]).to(device)
    ctx_y_t = torch.tensor(ctx_y_norm, dtype=torch.float32, device=device)

    preds = []
    with torch.no_grad():
        for start in range(0, len(candidate_graphs), batch_size):
            q_batch = Batch.from_data_list([to_pyg(g) for g in candidate_graphs[start : start + batch_size]]).to(device)
            pred = graph_pfn_model(ctx_batch, ctx_y_t, q_batch)
            preds.append(pred.detach().cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0).reshape(-1)
    return pred_norm * y_std + y_mean


def sample_yaw_candidates(env, obs0, n_candidates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_turbines = get_num_turbines(env, obs0)
    return rng.uniform(-40.0, 40.0, size=(n_candidates, n_turbines)).astype(np.float32)


def choose_with_surrogate(env, obs0, predict_fn, n_candidates: int, seed: int):
    candidates = sample_yaw_candidates(env, obs0, n_candidates, seed)
    scores = np.asarray(predict_fn(candidates)).reshape(-1)
    best_idx = int(np.argmax(scores))
    return candidates[best_idx], float(scores[best_idx])


def run_layout_seed_scenario(
    env_name: str,
    seed: int,
    scenario: ScenarioConfig,
    graph_pfn_model,
    args,
    device: torch.device,
) -> pd.DataFrame:
    env = make_env(env_name, args.max_steps)
    context = collect_context_data(env, seed, args.n_initial, scenario, args.max_steps)
    obs0 = context["obs0"]
    ws, wd = context["ws"], context["wd"]
    x_tab = context["x_tab"]
    y = context["y_reward"]
    graphs = context["graphs"]

    print(
        f"{scenario.name} | {env_name} | seed={seed} | "
        f"train_ws=({context['context_ws'].min():.2f}, {context['context_ws'].max():.2f}) | "
        f"test=({ws:.2f}, {wd:.2f})"
    )

    gp_model = fit_gp_model(x_tab, y)
    tab_model = SimpleTabPFNRegressor().fit(x_tab, y)

    def gp_predict(cand_yaws):
        x = np.stack([flatten_tabular_features(yw, env, ws, wd, obs0) for yw in cand_yaws], axis=0)
        return gp_model.predict(x)

    def tab_predict(cand_yaws):
        x = np.stack([flatten_tabular_features(yw, env, ws, wd, obs0) for yw in cand_yaws], axis=0)
        return tab_model.predict(x)

    def graph_predict(cand_yaws):
        q_graphs = [wfcrl_graph_from_yaws(env, yw, ws, wd, obs0) for yw in cand_yaws]
        return score_graphpfn_candidates(graph_pfn_model, graphs, y, q_graphs, device=device)

    yaw_do_nothing = np.zeros(get_num_turbines(env, obs0), dtype=np.float32)
    yaw_random = sample_yaw_candidates(env, obs0, 1, seed)[0]
    yaw_gp, pred_gp = choose_with_surrogate(env, obs0, gp_predict, args.n_candidates, seed)
    yaw_tab, pred_tab = choose_with_surrogate(env, obs0, tab_predict, args.n_candidates, seed + 1000)
    yaw_graph, pred_graph = choose_with_surrogate(env, obs0, graph_predict, args.n_candidates, seed + 2000)

    methods = {
        "Do-Nothing": (yaw_do_nothing, np.nan),
        "Random": (yaw_random, np.nan),
        "GP": (yaw_gp, pred_gp),
        tab_model.display_name: (yaw_tab, pred_tab),
        "GraphPFN": (yaw_graph, pred_graph),
    }

    rows = []
    for method, (yaws, pred) in methods.items():
        reward, power = ramp_and_eval(env, yaws, seed=seed, options=scenario.eval_options, max_steps=args.max_steps)
        rows.append(
            {
                "scenario": scenario.name,
                "layout": env_name,
                "seed": seed,
                "method": method,
                "reward": float(reward),
                "power": float(power),
                "predicted_reward": float(pred) if np.isfinite(pred) else np.nan,
                "test_wind_speed": float(ws),
                "test_wind_direction": float(wd),
                "train_wind_speed_min": float(context["context_ws"].min()),
                "train_wind_speed_max": float(context["context_ws"].max()),
                "train_wind_direction_min": float(context["context_wd"].min()),
                "train_wind_direction_max": float(context["context_wd"].max()),
                "n_turbines": int(get_num_turbines(env, obs0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        results[results["method"] == "Do-Nothing"][["scenario", "layout", "seed", "reward"]]
        .rename(columns={"reward": "baseline_reward"})
    )
    plot_df = results.merge(baseline, on=["scenario", "layout", "seed"], how="left")
    plot_df["reward_gain_vs_do_nothing"] = plot_df["reward"] - plot_df["baseline_reward"]
    return (
        plot_df.groupby(["scenario", "layout", "method"], as_index=False)
        .agg(
            mean_gain=("reward_gain_vs_do_nothing", "mean"),
            std_gain=("reward_gain_vs_do_nothing", "std"),
            mean_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            mean_power=("power", "mean"),
            std_power=("power", "std"),
        )
        .fillna(0.0)
        .sort_values(["scenario", "layout", "method"])
        .reset_index(drop=True)
    )


def rank_diagnostic(results: pd.DataFrame, out_dir: Path) -> None:
    # Small helper for quick sanity checks on result ordering in saved CSVs.
    method_counts = results.groupby(["scenario", "method"]).size().reset_index(name="count")
    method_counts.to_csv(out_dir / "method_counts.csv", index=False)


def parse_layouts(layout_args: list[str], max_layouts: int | None) -> list[str]:
    if not layout_args or layout_args == ["slide"]:
        layouts = SLIDE_LAYOUTS
    elif layout_args == ["fallback"]:
        layouts = FALLBACK_LAYOUTS
    else:
        layouts = layout_args
    if max_layouts:
        layouts = layouts[:max_layouts]
    return layouts


def validate_layouts(layouts: list[str], max_steps: int) -> list[str]:
    valid = []
    for layout in layouts:
        try:
            _ = make_env(layout, max_steps)
            valid.append(layout)
        except Exception as exc:
            print(f"Skipping invalid layout {layout}: {type(exc).__name__}: {exc}")
    if not valid:
        raise RuntimeError("No valid FLORIS layouts found.")
    return valid


def run(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    load_tabpfn_token_from_colab_secret()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("device:", device)
    print("wfcrl env count:", len(envs.list_envs()) if hasattr(envs, "list_envs") else "unknown")

    layouts = validate_layouts(parse_layouts(args.layouts, args.max_layouts), args.max_steps)
    scenarios = [
        ScenarioConfig("Scenario 1", eval_options=FIXED_FLORIS_TEST_OPTIONS, context_options=FIXED_FLORIS_TEST_OPTIONS),
        ScenarioConfig("Scenario 2", eval_options=FIXED_FLORIS_TEST_OPTIONS, context_sampler=sample_scenario2_options),
    ]

    graph_pfn_model = train_graphpfn(args.graph_train_steps, device=device, n_turbines=args.synthetic_turbines)

    all_runs, errors = [], []
    for layout in layouts:
        for seed in args.seeds:
            for scenario in scenarios:
                try:
                    all_runs.append(run_layout_seed_scenario(layout, seed, scenario, graph_pfn_model, args, device))
                except Exception as exc:
                    errors.append(
                        {
                            "scenario": scenario.name,
                            "layout": layout,
                            "seed": seed,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"FAILED {scenario.name} {layout} seed={seed}: {type(exc).__name__}: {exc}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "scenario_errors.csv", index=False)
    if not all_runs:
        raise RuntimeError("No successful scenario runs. See scenario_errors.csv.")

    results = pd.concat(all_runs, ignore_index=True)
    summary = summarize_results(results)
    results.to_csv(out_dir / "scenario_1_2_floris_results.csv", index=False)
    summary.to_csv(out_dir / "scenario_1_2_floris_summary.csv", index=False)
    rank_diagnostic(results, out_dir)
    print("wrote:", out_dir / "scenario_1_2_floris_results.csv")
    print("wrote:", out_dir / "scenario_1_2_floris_summary.csv")
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WFCRL GraphPFN/TabPFN on FLORIS Scenarios 1 and 2.")
    parser.add_argument("--layouts", nargs="*", default=["slide"], help="'slide', 'fallback', or explicit layout names.")
    parser.add_argument("--max-layouts", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-initial", type=int, default=32)
    parser.add_argument("--n-candidates", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--graph-train-steps", type=int, default=100)
    parser.add_argument("--synthetic-turbines", type=int, default=9)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
