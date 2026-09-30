"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
PUBLIC_ROOT = Path(__file__).resolve().parents[2]
HERE = PUBLIC_ROOT / 'outputs/figure_sources/f3'
REFIT_ROOT = PUBLIC_ROOT
FROZEN_IDS = ['D4_K10_L9_U47', 'D4_K9_L8_U49', 'D4_K6_L8_U50', 'D2_K9_L8_U50', 'D2_K8_L8_U39']
ASSIGNMENTS = ['D4_K10_L9_U47', 'D2_K9_L8_U50']
KEYS = ['participant_id_norm', 'hippo_id_norm']
COORDS = ['Dim0', 'Dim1', 'Dim2', 'Dim3']
EXPECTED_COUNTS = {'D4_K10_L9_U47': [26, 43, 9, 47, 47, 10, 18, 17, 17, 22], 'D4_K9_L8_U49': [49, 49, 8, 49, 17, 16, 31, 20, 17], 'D4_K6_L8_U50': [50, 46, 50, 50, 43, 17], 'D2_K9_L8_U50': [29, 49, 12, 40, 50, 32, 12, 18, 14], 'D2_K8_L8_U39': [37, 39, 31, 38, 39, 39, 15, 18]}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
ELLIPSE_SD = 1.5
EVENT_MARKER_AREA = 12
CENTROID_MARKER_AREA = 18
AXIS_WIDTH_IN = 4.15
BAR_WIDTH = 0.72
BAR_XLIM = (0.5, 10.5)
BAR_YLIM = (0, 55)

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def load_inputs():
    require((REFIT_ROOT / 'outputs/assignments').is_dir(), 'Missing computed basin assignments.')
    manifest = json.loads((HERE / 'sources.json').read_text())
    loaded = {}
    for role, info in manifest['files'].items():
        path = REFIT_ROOT / info['path_from_0_REFIT']
        require(path.is_file(), f'Missing input: {path}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], f"Canonical input differs from reviewed version: {info['path_from_0_REFIT']}")
        loaded[role] = json.loads(path.read_text()) if path.suffix == '.json' else pd.read_csv(path, keep_default_na=False)
    coords = loaded['coordinates'][KEYS + COORDS].copy()
    memberships = loaded['memberships']
    summary = loaded['summary'].set_index('assignment')
    require(list(summary.index) == FROZEN_IDS, 'The five frozen assignment IDs/order differ.')
    require(list(memberships.columns) == KEYS + FROZEN_IDS, 'Membership columns differ.')
    require(len(coords) == len(memberships) == 256, 'Expected 256 events in both inputs.')
    require(not coords.duplicated(KEYS).any() and (not memberships.duplicated(KEYS).any()), 'Duplicate participant/event key.')
    require(set(map(tuple, coords[KEYS].to_numpy())) == set(map(tuple, memberships[KEYS].to_numpy())), 'Coordinates and memberships describe different events.')
    events = coords.merge(memberships[KEYS + ASSIGNMENTS], on=KEYS, how='left', validate='one_to_one', sort=False)
    require(np.isfinite(events[COORDS].to_numpy(float)).all(), 'Nonfinite coordinates.')
    require(events.groupby(KEYS[0]).size().to_dict() == {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}, 'Participant event counts differ.')
    edges = loaded['edges']
    require(len(edges) == 251 and (not edges.duplicated().any()), 'Expected 251 unique original edges.')
    valid_keys = set(map(tuple, events[KEYS].to_numpy()))
    for endpoint in ('source_event', 'target_event'):
        require(set(zip(edges[KEYS[0]], edges[endpoint])).issubset(valid_keys), f'An edge {endpoint} is absent from the event table.')
    support_records = []
    for aid in ASSIGNMENTS:
        row = summary.loc[aid]
        meta = loaded[aid + '/metadata']
        require(meta['assignment_id'] == aid, 'Assignment metadata ID differs.')
        k = int(row.K)
        require(meta['K'] == k and meta['full_sample_events'] == 256 and (meta['full_sample_edges'] == 251), 'Assignment metadata size differs.')
        require(meta['fit_axes'] == ([0, 1, 2, 3] if row.dimension == 4 else [0, 3]), 'Assignment fit axes differ from the frozen definition.')
        require(set(events[aid]) == set(range(1, k + 1)), f'Unexpected basin labels for {aid}.')
        counts = events.groupby(aid).size().sort_index()
        require(counts.tolist() == EXPECTED_COUNTS[aid], f'Frozen basin counts differ for {aid}.')
        require(counts.min() == row.actual_minimum_events and counts.max() == row.actual_maximum_events, f'Assignment summary count range differs for {aid}.')
        source_labels = edges.merge(events[KEYS + [aid]], left_on=[KEYS[0], 'source_event'], right_on=KEYS, validate='many_to_one', how='left')
        outgoing = source_labels.groupby(aid).size().reindex(range(1, k + 1), fill_value=0)
        require(int(outgoing.min()) == int(row.minimum_outgoing), f'Minimum outgoing support differs for {aid}.')
        diagnostic = loaded[aid + '/diagnostics'].set_index('state').sort_index()
        centroids = loaded[aid + '/centroids'].set_index('state').sort_index()
        require(list(diagnostic.index) == list(centroids.index) == list(range(1, k + 1)), 'Missing or duplicate diagnostic/centroid state IDs.')
        means = events.groupby(aid)[COORDS].mean().sort_index()
        require(np.allclose(means, centroids[COORDS], rtol=0, atol=1e-12), f'Centroids do not match the supplied coordinate frame for {aid}.')
        for state, group in events.groupby(aid, sort=True):
            d = diagnostic.loc[state]
            participant_counts = group.groupby(KEYS[0]).size()
            require(int(d.n_events) == len(group) and int(d.outgoing) == int(outgoing.loc[state]), f'State support differs: {aid}, B{state}.')
            require(int(d.participants) == len(participant_counts) and np.isclose(d.max_participant_share, participant_counts.max() / len(group), atol=1e-12, rtol=0), f'Participant composition differs: {aid}, B{state}.')
            support_records.append({'assignment': aid, 'dimension': int(row.dimension), 'K': k, 'basin': int(state), 'events': len(group), 'outgoing': int(outgoing.loc[state]), 'participants': len(participant_counts), 'max_participant_share': float(d.max_participant_share), 'color': COLORS[int(state) - 1], **{c: float(means.loc[state, c]) for c in COORDS}})
    return (events, summary.loc[ASSIGNMENTS], pd.DataFrame(support_records), manifest)

def projection_limits(events, axes):
    xy = events[[f'Dim{i}' for i in axes]].to_numpy()
    lo, hi = (xy.min(axis=0), xy.max(axis=0))
    if axes == (0, 3):
        pad = (0.05 - 1.1 / 50) * (hi - lo)
    else:
        for aid in ASSIGNMENTS:
            for _, group in events.groupby(aid):
                values = group[[f'Dim{i}' for i in axes]].to_numpy(float)
                center = values.mean(axis=0)
                radius = ELLIPSE_SD * values.std(axis=0, ddof=1)
                lo = np.minimum(lo, center - radius)
                hi = np.maximum(hi, center + radius)
        pad = 0.075 * (hi - lo)
    return ((lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1]))

def add_covariance_ellipse(ax, xy, color):
    values, vectors = np.linalg.eigh(np.cov(xy, rowvar=False, ddof=1))
    values = np.maximum(values, 0)
    order = np.argsort(values)[::-1]
    values, vectors = (values[order], vectors[:, order])
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * ELLIPSE_SD * np.sqrt(values)
    if width > 1e-12 and height > 1e-12:
        ax.add_patch(Ellipse(xy.mean(axis=0), width=width, height=height, angle=angle, facecolor='none', edgecolor=color, linewidth=0.8, alpha=0.6, zorder=2))

def draw_map(ax, events, aid, axes, limits, title, letter=None):
    cols = [f'Dim{i}' for i in axes]
    for state, group in events.groupby(aid, sort=True):
        xy = group[cols].to_numpy(float)
        color = COLORS[int(state) - 1]
        add_covariance_ellipse(ax, xy, color)
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=EVENT_MARKER_AREA, alpha=0.75, edgecolors='none', zorder=3)
        center = xy.mean(axis=0)
        ax.scatter(*center, s=CENTROID_MARKER_AREA, marker='o', facecolors='#000000', edgecolors='none', linewidths=0, zorder=5)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(cols[0], labelpad=2)
    ax.set_ylabel(cols[1], labelpad=2)
    ax.set_title(title, loc='left', fontsize=10, fontweight='semibold', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=0.6, labelsize=8)
    if letter:
        ax.text(-0.15, 1.075, letter, transform=ax.transAxes, fontsize=12, fontweight='bold')

def overlap_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

def add_basin_labels(fig, ax, events, aid, axes, font):
    means = events.groupby(aid)[[f'Dim{i}' for i in axes]].mean().sort_index()
    pixel_centers = ax.transData.transform(means.to_numpy())
    renderer = fig.canvas.get_renderer()
    factor = fig.dpi / 72
    bounds = ax.get_window_extent()
    placed = []
    records = []
    offsets = [(x, y) for radius in (10, 17, 25, 34, 44) for x, y in ((radius, radius), (radius, -radius), (-radius, radius), (-radius, -radius), (radius, 0), (-radius, 0), (0, radius), (0, -radius))]
    for idx, (basin, center) in enumerate(means.iterrows()):
        text = f'B{basin}'
        width, height, _ = renderer.get_text_width_height_descent(text, FontProperties(family=font, size=8), False)
        width += 5 * factor
        height += 6 * factor
        origin = pixel_centers[idx]
        best = None
        for dx, dy in offsets:
            pos = origin + np.array([dx, dy]) * factor
            box = (pos[0] - width / 2, pos[1] - height / 2, pos[0] + width / 2, pos[1] + height / 2)
            outside = max(0, bounds.x0 + 2 - box[0]) + max(0, bounds.y0 + 2 - box[1]) + max(0, box[2] - bounds.x1 + 2) + max(0, box[3] - bounds.y1 + 2)
            collision = sum((overlap_area(box, old) for old in placed))
            marker_hit = sum((overlap_area(box, (p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4)) for p in pixel_centers))
            score = 10000 * outside + 100 * collision + 40 * marker_hit + dx * dx + dy * dy
            if best is None or score < best[0]:
                best = (score, dx, dy, box)
        _, dx, dy, box = best
        placed.append(box)
        ax.annotate(text, center.to_numpy(), xytext=(dx, dy), textcoords='offset points', ha='center', va='center', fontsize=8, color='#171717', zorder=7, bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.84), arrowprops=dict(arrowstyle='-', color='#555555', linewidth=0.45, shrinkA=1, shrinkB=3))
        records.append({'assignment': aid, 'projection': f'Dim{axes[0]}xDim{axes[1]}', 'basin': int(basin), 'label_offset_x_points': dx, 'label_offset_y_points': dy})
    return records

def draw_counts(ax, support, aid, title, letter=None):
    data = support.loc[support.assignment == aid].sort_values('basin')
    bars = ax.bar(data.basin, data.events, color=data.color, width=BAR_WIDTH, edgecolor='#000000', linewidth=0.7, zorder=3)
    ax.set_xlim(*BAR_XLIM)
    ax.set_ylim(*BAR_YLIM)
    ax.set_xticks(data.basin, [f'B{i}' for i in data.basin])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel('Basin-state', labelpad=4)
    ax.set_ylabel('Events', labelpad=4)
    ax.set_title(title, loc='left', fontsize=10, fontweight='semibold', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, labelsize=8, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e2e2e2', linewidth=0.45)
    if letter:
        ax.text(-0.15, 1.12, letter, transform=ax.transAxes, fontsize=12, fontweight='bold')
    return bars
