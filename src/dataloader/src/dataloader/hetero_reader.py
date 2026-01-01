"""
Utilities for loading ICU heterograph data into torch_geometric HeteroData.

Expected directory layout (configurable via config entries):

hetero_dir/
  nodes/
    stay_x.npy
    stay_feature_names.json
    stay_y.npy
    stay_ids.npy
    patient_x.npy
    patient_feature_names.json
    ...
  edges/
    patient_HAS_STAY_stay_edge_index.npy
    stay_HAS_PROC_procedure_edge_index.npy
    stay_HAS_PROC_procedure_edge_attr.npy
    stay_HAS_PROC_procedure_edge_attr_name.json
    ...

Each edge_index file should be shaped [2, E] or [E, 2].
Each edge_attr file should be shaped [E, D].
Feature name files are JSON objects mapping feature group names to sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


@dataclass
class NodeSpec:
    x: Path
    feature_names: Optional[Union[Path, dict]] = None
    y: Optional[Path] = None
    ids: Optional[Path] = None


@dataclass
class EdgeSpec:
    edge_index: Path
    edge_attr: Optional[Path] = None
    edge_attr_name: Optional[Union[Path, dict]] = None


def _load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_metadata(value: Optional[Union[Path, dict]]) -> Optional[dict]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return _load_json(value)


def _load_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected .npy array at {path}, got .npz archive.")
    return arr


def _to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(array).to(dtype=dtype)


def _load_edge_index(path: Path) -> torch.Tensor:
    edge_index = _load_array(path)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {edge_index.shape} at {path}")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, E] (or [E, 2]), got {edge_index.shape}"
        )
    return _to_tensor(edge_index, dtype=torch.long)


def build_heterodata(
    node_specs: Dict[str, NodeSpec],
    edge_specs: Dict[Tuple[str, str, str], EdgeSpec],
) -> HeteroData:
    data = HeteroData()

    for node_type, spec in node_specs.items():
        x = _to_tensor(_load_array(spec.x), dtype=torch.float)
        data[node_type].x = x

        feature_names = _resolve_metadata(spec.feature_names)
        if feature_names is not None:
            data[node_type].feature_names = feature_names

        if spec.y is not None:
            data[node_type].y = _to_tensor(_load_array(spec.y), dtype=torch.float)

        if spec.ids is not None:
            ids = _load_array(spec.ids)
            if ids.ndim != 1:
                raise ValueError(f"ids must be 1D for node type {node_type}")
            data[node_type].ids = _to_tensor(ids, dtype=torch.long)

    for edge_type, spec in edge_specs.items():
        edge_index = _load_edge_index(spec.edge_index)
        data[edge_type].edge_index = edge_index

        if spec.edge_attr is not None:
            edge_attr = _to_tensor(_load_array(spec.edge_attr), dtype=torch.float)
            data[edge_type].edge_attr = edge_attr

        edge_attr_name = _resolve_metadata(spec.edge_attr_name)
        if edge_attr_name is not None:
            data[edge_type].edge_attr_name = edge_attr_name

    return data


def load_hetero_spec(path: Path) -> Tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if "nodes" not in spec or "edges" not in spec:
        raise ValueError("hetero spec must include 'nodes' and 'edges' keys")
    return spec["nodes"], spec["edges"]


def _normalize_edge_key(edge_key) -> Tuple[str, str, str]:
    if isinstance(edge_key, (list, tuple)) and len(edge_key) == 3:
        return tuple(edge_key)
    if isinstance(edge_key, str):
        for sep in ("|", ",", "->", ":"):
            if sep in edge_key:
                parts = [part.strip() for part in edge_key.split(sep) if part.strip()]
                if len(parts) == 3:
                    return tuple(parts)
    raise ValueError(f"Edge key must be a 3-tuple/list or parseable string, got {edge_key!r}")


def validate_expected_schema(
    data: HeteroData,
    expected_nodes: Optional[Tuple[str, ...]] = None,
    expected_edges: Optional[Tuple[Tuple[str, str, str], ...]] = None,
) -> None:
    missing_nodes = []
    missing_edges = []

    if expected_nodes:
        for node_type in expected_nodes:
            if node_type not in data.node_types:
                missing_nodes.append(node_type)

    if expected_edges:
        for edge_type in expected_edges:
            if edge_type not in data.edge_types:
                missing_edges.append(edge_type)

    if missing_nodes or missing_edges:
        parts = []
        if missing_nodes:
            parts.append(f"missing node types: {missing_nodes}")
        if missing_edges:
            parts.append(f"missing edge types: {missing_edges}")
        raise ValueError("HeteroData schema mismatch: "  "; ".join(parts))


class HeteroGraphDataset(Dataset):
    """
    Dataset wrapper that yields a single HeteroData object built from files.
    """

    def __init__(self, config: dict):
        super().__init__()
        hetero_dir = Path(config["hetero_dir"])

        if "hetero_spec" in config:
            nodes_cfg, edges_cfg = load_hetero_spec(Path(config["hetero_spec"]))
        else:
            nodes_cfg = config["hetero_nodes"]
            edges_cfg = config["hetero_edges"]

        node_specs = {}
        if isinstance(nodes_cfg, list):
            node_items = nodes_cfg
        else:
            node_items = [{"type": key, **value} for key, value in nodes_cfg.items()]

        for node_cfg in node_items:
            node_type = node_cfg["type"]
            feature_names = node_cfg.get("feature_names")
            if isinstance(feature_names, str):
                feature_names = hetero_dir / feature_names
            node_specs[node_type] = NodeSpec(
                x=hetero_dir / node_cfg["x"],
                feature_names=feature_names,
                y=(hetero_dir / node_cfg["y"]) if node_cfg.get("y") else None,
                ids=(hetero_dir / node_cfg["ids"]) if node_cfg.get("ids") else None,
            )

        edge_specs = {}
        if isinstance(edges_cfg, list):
            edge_items = edges_cfg
        else:
            edge_items = []
            for edge_key, edge_cfg in edges_cfg.items():
                edge_items.append({"key": edge_key, **edge_cfg})

        for edge_cfg in edge_items:
            edge_key = edge_cfg.get("key") or edge_cfg.get("edge_type")
            if edge_key is None:
                edge_key = (edge_cfg["src"], edge_cfg["rel"], edge_cfg["dst"])
            src, rel, dst = _normalize_edge_key(edge_key)
            edge_attr_name = edge_cfg.get("edge_attr_name")
            if isinstance(edge_attr_name, str):
                edge_attr_name = hetero_dir / edge_attr_name
            edge_specs[(src, rel, dst)] = EdgeSpec(
                edge_index=hetero_dir / edge_cfg["edge_index"],
                edge_attr=(hetero_dir / edge_cfg["edge_attr"])
                if edge_cfg.get("edge_attr")
                else None,
                edge_attr_name=edge_attr_name,
            )

        self.data = build_heterodata(node_specs, edge_specs)

        expected_nodes = config.get("expected_node_types")
        expected_edges = config.get("expected_edge_types")
        if expected_nodes or expected_edges:
            expected_nodes = tuple(expected_nodes) if expected_nodes else None
            if expected_edges:
                expected_edges = tuple(_normalize_edge_key(edge) for edge in expected_edges)
            validate_expected_schema(self.data, expected_nodes, expected_edges)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> HeteroData:
        return self.data
