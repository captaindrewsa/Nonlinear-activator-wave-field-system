# -*- coding: utf-8 -*-
"""
run_discovery.py
================
Общий слой поиска run_dir внутри дерева sim_framework_v1.py.

Архитектура фреймворка:
    out_dir/
        <run_id_1>/
            config.json
            snapshots_2d/
                snapshot2d_<run_id_1>_t########.npz
                snapshot2d_<run_id_1>_t########_final.npz
            track.csv
            result.json
            processed/                <- создаётся анализаторами (17/18/npz_to_csv_gif)
                locales/
                lifecycle/
                media/
        <run_id_2>/
            ...

Правило: анализаторы НИКОГДА не пишут файлы рядом со snapshots_2d или
config.json. Всё новое уходит в run_dir/processed/<tool_name>/.

Этот модуль не имеет внешних зависимостей кроме stdlib и предоставляет:
    - is_run_dir(path)          -> bool
    - find_run_dirs(root)       -> List[str]
    - resolve_targets(args)     -> List[str]   (--run-id / --out-dir universal resolver)
    - processed_dir(run_dir, tool_name) -> str  (создаёт и возвращает путь)
    - load_run_config(run_dir)  -> dict | None
    - snapshot_dir_of(run_dir)  -> str
"""
from __future__ import annotations

import os
import json
from typing import List, Optional, Dict, Any


def norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def is_run_dir(path: str) -> bool:
    """
    run_dir определяется наличием config.json (сохраняется Simulator.__init__)
    и/или подпапки snapshots_2d.
    """
    if not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_snap = os.path.isdir(os.path.join(path, "snapshots_2d"))
    return has_config or has_snap


def find_run_dirs(root: str) -> List[str]:
    """
    Рекурсивно находит все run_dir внутри root (это может быть либо
    сам out_dir с серией прогонов, либо произвольная папка-контейнер).
    Если root сам является run_dir — возвращает [root].
    """
    root = norm_path(root)
    if is_run_dir(root):
        return [root]

    found = []
    for dirpath, dirnames, _filenames in os.walk(root):
        # не спускаемся внутрь processed/ и snapshots_2d/ в поисках run_dir
        dirnames[:] = [d for d in dirnames if d not in ("processed", "snapshots_2d", "seed_cache")]
        if is_run_dir(dirpath):
            found.append(norm_path(dirpath))
    return sorted(found)


def snapshot_dir_of(run_dir: str) -> str:
    cand = os.path.join(run_dir, "snapshots_2d")
    if os.path.isdir(cand):
        return cand
    # обратная совместимость со старым layout (snapshots_2d/<tag>)
    return run_dir


def load_run_config(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_id_of(run_dir: str) -> str:
    cfg = load_run_config(run_dir)
    if cfg and cfg.get("run_id"):
        return cfg["run_id"]
    return os.path.basename(norm_path(run_dir))


def processed_dir(run_dir: str, tool_name: str) -> str:
    """
    Возвращает (и создаёт) run_dir/processed/<tool_name>/.
    Все новые файлы анализаторов должны писаться только сюда.
    """
    path = os.path.join(run_dir, "processed", tool_name)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_targets(run_id: Optional[str], out_dir: Optional[str]) -> List[str]:
    """
    Универсальный резолвер точки входа для CLI всех анализаторов:
      --run-id PATH   -> ровно один run_dir (path указывает прямо на него)
      --out-dir PATH  -> все run_dir, найденные рекурсивно внутри

    Если задано и то и другое — run_id имеет приоритет.
    """
    if run_id:
        rd = norm_path(run_id)
        if not is_run_dir(rd):
            raise FileNotFoundError(
                f"--run-id указывает не на run_dir (нет config.json/snapshots_2d): {rd}"
            )
        return [rd]

    if out_dir:
        dirs = find_run_dirs(out_dir)
        if not dirs:
            raise FileNotFoundError(f"Внутри --out-dir не найдено ни одного run_dir: {out_dir}")
        return dirs

    raise ValueError("Нужно указать --run-id или --out-dir")
