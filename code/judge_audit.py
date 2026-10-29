from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
AXES = ['completeness', 'safety', 'sop_applicability', 'no_uncertainty', 'hallucination_severity']

def _family_of(model: str) -> str:
    if model.startswith('GigaChat-'):
        return 'gigachat'
    if model.startswith('yandexgpt'):
        return 'yandexgpt'
    if model.startswith('gpt-oss'):
        return 'gpt-oss'
    m = model.lower()
    if 'deepseek' in m:
        return 'deepseek'
    if 'qwen' in m:
        return 'qwen'
    return 'unknown'

def _match_type(judge_key: str, candidate_provider: str, candidate_model: str) -> str:
    """`self` if judge == candidate; `family` if same model lineage; else `unrelated`."""
    cand_key = f'{candidate_provider}:{candidate_model}'
    if judge_key == cand_key:
        return 'self'
    judge_model = judge_key.split(':', 1)[1] if ':' in judge_key else judge_key
    if _family_of(judge_model) == _family_of(candidate_model):
        return 'family'
    return 'unrelated'

def _avg_rank(values: list[float]) -> list[float]:
    """Averaged ranks (1-indexed, higher value = lower rank number i.e.
    closer to 1 = best)."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

def _spearman_from_ranks(ra: list[float], rb: list[float]) -> float | None:
    if not ra or len(ra) != len(rb):
        return None
    n = len(ra)
    mra = sum(ra) / n
    mrb = sum(rb) / n
    num = sum(((a - mra) * (b - mrb) for a, b in zip(ra, rb)))
    dxa = sum(((a - mra) ** 2 for a in ra))
    dxb = sum(((b - mrb) ** 2 for b in rb))
    denom = (dxa * dxb) ** 0.5
    if denom < 1e-12:
        return None
    return num / denom

def compute_self_preference(records: list[dict]) -> tuple[list[dict], list[str], set[str]]:
    """For each (judge, candidate, mode, axis): mean score by judge +
    Δ vs other judges on the same cell.

    Returns (per_pair_rows, judges_sorted, candidate_keys).
    """
    bucket: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    judges_set: set[str] = set()
    candidate_keys: set[str] = set()
    for r in records:
        if r.get('item_type') != 'open':
            continue
        ind = r.get('open_scores_individual') or {}
        if not ind:
            continue
        cell = (r['provider'], r['model'], r['mode'], r['item_id'])
        candidate_keys.add(f"{r['provider']}:{r['model']}")
        for judge, axis_dict in ind.items():
            judges_set.add(judge)
            if not isinstance(axis_dict, dict):
                continue
            for axis in AXES:
                v = axis_dict.get(axis)
                if v is None:
                    continue
                try:
                    bucket[cell][axis][judge] = float(v)
                except (TypeError, ValueError):
                    continue
    judges = sorted(judges_set)
    rows: list[dict] = []
    by_candidate: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for cell in bucket:
        by_candidate[cell[0], cell[1]].append(cell)
    for (cand_prov, cand_model), cells in by_candidate.items():
        for judge in judges:
            for axis in AXES:
                this_scores: list[float] = []
                gaps: list[float] = []
                n_with_others = 0
                for cell in cells:
                    axis_scores = bucket[cell].get(axis, {})
                    if judge not in axis_scores:
                        continue
                    this_scores.append(axis_scores[judge])
                    others = [v for j, v in axis_scores.items() if j != judge]
                    if others:
                        gaps.append(axis_scores[judge] - mean(others))
                        n_with_others += 1
                if not this_scores:
                    continue
                match = _match_type(judge, cand_prov, cand_model)
                rows.append({'judge': judge, 'candidate_provider': cand_prov, 'candidate_model': cand_model, 'axis': axis, 'match_type': match, 'n_cells': len(this_scores), 'judge_mean_score': round(mean(this_scores), 3), 'n_with_others': n_with_others, 'mean_gap_vs_others': round(mean(gaps), 3) if gaps else None})
    return (rows, judges, candidate_keys)

def compute_loo_rankings(records: list[dict], judges: list[str]) -> dict:
    """Per axis: baseline ranking (all judges) + LOO ranking (drop one).
    Returns {axis: {baseline: [...], loo: {judge: [...]}, spearman: {judge: ρ}}}.
    """
    bucket: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    candidates_set: set[tuple[str, str]] = set()
    for r in records:
        if r.get('item_type') != 'open':
            continue
        ind = r.get('open_scores_individual') or {}
        if not ind:
            continue
        cell = (r['provider'], r['model'], r['mode'], r['item_id'])
        candidates_set.add((r['provider'], r['model']))
        for judge, axis_dict in ind.items():
            if not isinstance(axis_dict, dict):
                continue
            for axis in AXES:
                v = axis_dict.get(axis)
                if v is None:
                    continue
                try:
                    bucket[cell][axis][judge] = float(v)
                except (TypeError, ValueError):
                    continue
    candidates = sorted(candidates_set)

    def model_mean(cells_for_model, axis, allowed_judges):
        vals: list[float] = []
        for cell in cells_for_model:
            scores = {j: v for j, v in bucket[cell].get(axis, {}).items() if j in allowed_judges}
            if scores:
                vals.append(mean(scores.values()))
        return (mean(vals) if vals else None, len(vals))
    cells_by_cand: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for cell in bucket:
        cells_by_cand[cell[0], cell[1]].append(cell)
    result: dict[str, dict] = {}
    for axis in AXES:
        baseline_vals = []
        baseline_n = []
        for cand in candidates:
            v, n = model_mean(cells_by_cand[cand], axis, set(judges))
            baseline_vals.append(v if v is not None else float('nan'))
            baseline_n.append(n)
        loo_raw: dict[str, dict] = {}
        for drop in judges:
            allowed = set(judges) - {drop}
            loo_vals = []
            loo_n = []
            for cand in candidates:
                v, n = model_mean(cells_by_cand[cand], axis, allowed)
                loo_vals.append(v if v is not None else float('nan'))
                loo_n.append(n)
            loo_raw[drop] = {'vals': loo_vals, 'n': loo_n}
        baseline_ranks_aligned: dict[str, list[float]] = {}
        for drop in judges:
            both_indices = [i for i in range(len(candidates)) if baseline_vals[i] == baseline_vals[i] and loo_raw[drop]['vals'][i] == loo_raw[drop]['vals'][i]]
            base_subset = [baseline_vals[i] for i in both_indices]
            loo_subset = [loo_raw[drop]['vals'][i] for i in both_indices]
            base_ranks = _avg_rank(base_subset) if base_subset else []
            loo_ranks_subset = _avg_rank(loo_subset) if loo_subset else []
            rho = _spearman_from_ranks(base_ranks, loo_ranks_subset)
            full_loo_ranks = [float('nan')] * len(candidates)
            for sub_idx, orig_idx in enumerate(both_indices):
                full_loo_ranks[orig_idx] = loo_ranks_subset[sub_idx]
            loo_raw[drop].update({'ranks': full_loo_ranks, 'spearman': round(rho, 3) if rho is not None else None, 'n_used_for_rho': len(both_indices), 'n_dropped_from_rho': len(candidates) - len(both_indices)})
        full_baseline_ranks = _avg_rank([baseline_vals[i] for i in range(len(candidates)) if baseline_vals[i] == baseline_vals[i]])
        full_baseline_ranks_aligned: list[float] = []
        rank_iter = iter(full_baseline_ranks)
        for v in baseline_vals:
            full_baseline_ranks_aligned.append(next(rank_iter) if v == v else float('nan'))
        result[axis] = {'candidates': [f'{p}:{m}' for p, m in candidates], 'baseline_vals': baseline_vals, 'baseline_n': baseline_n, 'baseline_ranks': full_baseline_ranks_aligned, 'loo': loo_raw}
    return result
