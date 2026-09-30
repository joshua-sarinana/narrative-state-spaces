"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
PUBLIC_ROOT = Path(__file__).resolve().parents[2]
HERE = PUBLIC_ROOT / 'outputs/figure_sources/f5'
AIDS = ['D4_K10_L9_U47', 'D2_K9_L8_U50']
SELECTED = AIDS.copy()
LABELS = dict(zip(AIDS, ['4-D 10-Basin', '2-D 9-Basin']))
ACCENT = {a: '#22689E' if a.startswith('D4') else '#BA6816' for a in AIDS}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
KEYS = ['participant_id_norm', 'hippo_id_norm']
PROFILES = ['P', 'I_plus_gammaP', 'M', 'M_minus_I']
PROFILE_LABELS = ['P', 'I + γP', 'M', 'M − I']
GAMMA = 0.9

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def read(path, index=False):
    return pd.read_csv(path, index_col=0 if index else None, float_precision='round_trip', keep_default_na=bool(index))

def close(actual, expected, message, tolerance=2e-12):
    require(np.allclose(np.asarray(actual, float), np.asarray(expected, float), atol=tolerance, rtol=0, equal_nan=True), message)

def distances(values):
    v = np.asarray(values, float)
    return np.sqrt(np.sum((v[:, None, :] - v[None, :, :]) ** 2, axis=2))

def normalize(values):
    v = np.asarray(values, float)
    total = v.sum(1, keepdims=True)
    return np.divide(v, total, out=np.zeros_like(v), where=total > 0)

def row(frame, **filters):
    for key, value in filters.items():
        frame = frame[frame[key] == value]
    require(len(frame) == 1, f'Expected one saved result: {filters}')
    return frame.iloc[0]

def check_successor(p, m, message):
    k = len(p)
    close((np.eye(k) - GAMMA * np.asarray(p)) @ np.asarray(m), np.eye(k), message, tolerance=1e-10)
    require(np.isfinite(m).all().all() and (np.asarray(m) >= -1e-12).all(), message + ' (nonfinite or negative)')

def check_p(null, observed, expected, tail='two_sided'):
    null = np.asarray(null, float)
    require(len(null) == 10000 and np.isfinite(null).all(), 'Expected 10,000 finite saved null draws.')
    if tail == 'two_sided':
        hits = np.abs(null) >= abs(observed)
    elif tail == 'lower':
        hits = null <= observed
    else:
        hits = null >= observed
    close((1 + hits.sum()) / (len(null) + 1), expected, 'Saved p-value differs from saved null draws.')

def load_inputs():
    manifest = json.loads((HERE / 'inputs/sources.json').read_text())
    for rel, info in manifest['files'].items():
        p = HERE / 'inputs' / rel
        require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == info['sha256'], f'Bundled source changed or missing: {rel}')
    project_root = PUBLIC_ROOT
    manifest['local_canonical_files_verified'] = 0
    for rel, info in manifest['files'].items():
        canonical = info.get('canonical_path_from_0_REFIT')
        if canonical and (project_root / canonical).is_file():
            require(hashlib.sha256((project_root / canonical).read_bytes()).hexdigest() == info['sha256'], f'Local canonical source differs: {canonical}. Review source versions before plotting.')
            manifest['local_canonical_files_verified'] += 1
    frozen = read(HERE / 'inputs/frozen_memberships.csv')
    profiles = read(HERE / 'inputs/comparisons/profile_definitions.csv')
    orders = read(HERE / 'inputs/comparisons/event_order.csv')
    four = read(HERE / 'inputs/comparisons/four_conditions_geometry.csv')
    require(set(AIDS) <= set(profiles.assignment_id) and set(AIDS) <= set(orders.assignment_id), 'Frozen control assignments differ.')
    profiles = profiles[(profiles.geometry_D == 4) & profiles.assignment_id.isin(AIDS)].copy()
    orders = orders[(orders.geometry_D == 4) & orders.assignment_id.isin(AIDS)].copy()
    require(len(profiles) == 8 and len(orders) == 4, 'Expected eight selected D4 profile and four order results.')
    require((profiles.status == 'ok').all() and (orders.status == 'ok').all(), 'A control is not estimable.')
    data = {}
    audit = []
    for aid in AIDS:
        root = HERE / 'inputs' / aid
        base = root / 'baseline'
        d = {'aid': aid, 'events': read(base / 'event_memberships_features_coordinates.csv'), 'edges': read(base / 'event_transition_records.csv'), 'P': read(base / 'transition_probabilities_empirical.csv', True), 'counts': read(base / 'transition_counts.csv', True), 'centroids': read(base / 'centroids_D4.csv', True)}
        ev, edges = (d['events'], d['edges'])
        k = len(d['P'])
        labels = [f'B{i}' for i in range(1, k + 1)]
        iu = np.triu_indices(k, 1)
        d.update(k=k, labels=labels, iu=iu, geometry_distances=distances(d['centroids']))
        require(len(ev) == 256 and len(edges) == 251 and (ev.participant_id_norm.nunique() == 5), 'Expected 256 events, 251 transitions and five participants.')
        require(not ev.duplicated(KEYS).any(), 'Duplicate event keys.')
        match = ev[KEYS + ['state']].merge(frozen[KEYS + [aid]], on=KEYS, validate='one_to_one')
        require(len(match) == 256 and np.array_equal(match.state, match[aid]), 'Frozen membership mismatch.')
        lookup = ev.set_index(KEYS)
        for end in ['source', 'target']:
            keys = pd.MultiIndex.from_frame(edges[['participant_id_norm', end + '_event']])
            require(keys.isin(lookup.index).all(), 'Unresolved event edge.')
            require(np.array_equal(lookup.loc[keys, 'state'], edges[end + '_state']), 'Transition labels differ from memberships.')
        counts = np.zeros((k, k))
        np.add.at(counts, (edges.source_state.to_numpy(int) - 1, edges.target_state.to_numpy(int) - 1), 1)
        close(counts, d['counts'], 'Saved event counts differ from frozen event edges.')
        close(normalize(counts), d['P'], 'Saved event probabilities differ.')
        require(list(d['P'].index) == list(d['P'].columns) == list(d['centroids'].index) == labels, 'Basin order mismatch.')
        close(ev.groupby('state')[[f'Dim{i}' for i in range(4)]].mean(), d['centroids'], 'Full-event centroids differ.')
        x = d['geometry_distances'][iu]
        d['geometry_pairs'] = x
        pr = root / 'profiles'
        component = read(pr / 'profile_correspondence.csv')
        nulls = read(pr / 'permutation_nulls.csv')
        d['profile_matrices'] = {p: read(pr / f'profile_{p}.csv', True) for p in PROFILES}
        close(d['profile_matrices']['P'], d['P'], 'P profile differs from empirical P.')
        close(d['profile_matrices']['I_plus_gammaP'], np.eye(k) + GAMMA * d['P'], 'One-step profile differs.')
        d['M'] = d['profile_matrices']['M']
        check_successor(d['P'], d['M'], 'Saved successor equation fails.')
        close(d['profile_matrices']['M_minus_I'], d['M'] - np.eye(k), 'Identity-removed profile differs.')
        for p in PROFILES:
            a = row(profiles, assignment_id=aid, profile=p)
            c = row(component, geometry_D=4, profile=p)
            close(a[['pearson_r', 'permutation_p']], c[['pearson_r', 'permutation_p']], 'Profile aggregate differs from component.')
            close(np.corrcoef(x, distances(d['profile_matrices'][p])[iu])[0, 1], c.pearson_r, 'Profile distance correlation differs.')
            check_p(nulls['D4_' + p], c.pearson_r, c.permutation_p)
            require(c.gamma == GAMMA and c.n_permutations == 10000 and (c.seed == 42), 'Profile settings differ.')
        folder = root / 'order'
        component = read(folder / 'order_correspondence.csv')
        nulls = read(folder / 'permutation_nulls.csv')
        close(read(folder / 'narrated_counts.csv', True), counts, 'Narrated counts differ.')
        close(read(folder / 'reversed_counts.csv', True), counts.T, 'Reversed counts are not the narrated transpose.')
        for order in ['narrated', 'reversed']:
            p = read(folder / f'{order}_probabilities.csv', True)
            m = read(folder / f'{order}_successor.csv', True)
            close(p, normalize(counts if order == 'narrated' else counts.T), 'Order row normalization differs.')
            check_successor(p, m, 'Order successor equation fails.')
            a = row(orders, assignment_id=aid, order=order)
            c = row(component, geometry_D=4, order=order)
            close(a[['pearson_r', 'label_permutation_p_two_sided', 'order_shuffle_p_upper_tail']], c[['pearson_r', 'label_permutation_p_two_sided', 'order_shuffle_p_upper_tail']], 'Order aggregate differs.')
            close(np.corrcoef(x, distances(m)[iu])[0, 1], c.pearson_r, 'Order correlation differs.')
            check_p(nulls[order + '_D4_label_r'], c.pearson_r, c.label_permutation_p_two_sided)
            check_p(nulls['D4_order_shuffle_r'], c.pearson_r, c.order_shuffle_p_upper_tail, 'upper')
            close(np.quantile(nulls.D4_order_shuffle_r, [0.025, 0.975]), [c.shuffle_null_q025, c.shuffle_null_q975], 'Order-null interval differs.')
            close(nulls.D4_order_shuffle_r.mean(), c.shuffle_null_mean_r, 'Order-null mean differs.')
        d['event_stat'] = row(profiles, assignment_id=aid, profile='M')
        if aid in SELECTED:
            folder = root / 'geometry'
            d['pca'] = read(folder / 'successor_pca_coordinates.csv', True)
            d['variance'] = read(folder / 'successor_pca_variance.csv')
            d['correspondence'] = read(folder / 'distance_correspondence.csv')
            d['diagnostics'] = read(folder / 'geometry_diagnostics.csv').iloc[0]
            d['aligned'] = read(folder / 'procrustes_aligned_coordinates.csv')
            d['pairs'] = read(folder / 'pairs_D4_full_profile.csv')
            savedm = read(folder / 'successor_matrix.csv', True)
            nulls = read(folder / 'permutation_nulls.csv')
            close(savedm, d['M'], 'Successor matrix differs between saved components.')
            close(read(folder / 'geometry_distances_D4.csv', True), d['geometry_distances'], 'Saved D4 distances differ.')
            require(list(d['pca'].index) == labels and list(d['aligned'].basin) == labels, 'PCA or alignment labels differ.')
            require(list(zip(d['pairs'].basin_a, d['pairs'].basin_b)) == [(labels[i], labels[j]) for i, j in zip(*iu)], 'Pair labels differ.')
            close(d['pairs'].geometry_distance, x, 'Saved geometry pair distances differ.')
            close(d['pairs'].predictive_distance, distances(savedm)[iu], 'Saved complete-profile distances differ.')
            scores = d['pca'].to_numpy()
            centered = savedm.to_numpy() - savedm.to_numpy().mean(axis=0)
            close(distances(savedm), distances(scores), 'Complete PCA coordinates do not preserve row distances.', 1e-10)
            close(scores.mean(axis=0), np.zeros(k - 1), 'Saved PCA scores are not centered.')
            close(scores.T @ scores, np.diag(np.diag(scores.T @ scores)), 'PCA score columns are not orthogonal.', 1e-10)
            close((scores * scores).sum(0) / (centered * centered).sum(), d['variance'].variance_ratio, 'Saved PCA variance differs.')
            close(d['variance'].variance_ratio.cumsum(), d['variance'].cumulative_variance_ratio, 'Cumulative variance differs.')
            for rep, values in [('full_profile', savedm.to_numpy())] + [(f'PCA{i}', scores[:, :i]) for i in range(1, k)]:
                stat = row(d['correspondence'], geometry_D=4, representation=rep)
                close(np.corrcoef(x, distances(values)[iu])[0, 1], stat.pearson_r, 'PCA truncation correlation differs.')
                check_p(nulls['D4_' + rep], stat.pearson_r, stat.two_sided_common_label_permutation_p)
            c = d['centroids'][['Dim0', 'Dim3']].to_numpy(copy=True)
            c -= c.mean(0)
            c /= np.linalg.norm(c)
            a = d['aligned']
            close(c, a[['terrain_x', 'terrain_y']], 'Procrustes terrain coordinates differ.')
            z = scores[:, :2] - scores[:, :2].mean(0)
            z /= np.linalg.norm(z)
            close(distances(z), distances(a[['predictive_x', 'predictive_y']]), 'Aligned successor plane changes normalized distances.')
            close(np.sum((a[['terrain_x', 'terrain_y']].to_numpy() - a[['predictive_x', 'predictive_y']].to_numpy()) ** 2), d['diagnostics'].procrustes_disparity, 'Procrustes disparity differs.')
            check_p(nulls.procrustes_disparity, d['diagnostics'].procrustes_disparity, d['diagnostics'].lower_tail_permutation_p, 'lower')
            v = root / 'visit'
            d['visit_counts'] = read(v / 'visit_transition_counts.csv', True)
            d['visit_P'] = read(v / 'visit_transition_probabilities.csv', True)
            d['visit_M'] = read(v / 'visit_successor_matrix.csv', True)
            expected = counts.copy()
            np.fill_diagonal(expected, 0)
            close(d['visit_counts'], expected, 'Visit collapse differs from diagonal removal at count level.')
            close(d['visit_P'], normalize(expected), 'Visit transition probabilities differ.')
            check_successor(d['visit_P'], d['visit_M'], 'Visit successor equation fails.')
            d['visit_stat'] = row(read(v / 'visit_geometry_correspondence.csv'), geometry_D=4)
            y = distances(d['visit_M'])[iu]
            d['visit_pairs'] = d['pairs'][['basin_a', 'basin_b', 'geometry_distance']].copy()
            d['visit_pairs']['predictive_distance'] = y
            close(np.corrcoef(x, y)[0, 1], d['visit_stat'].pearson_r, 'Visit correlation differs.')
            check_p(read(v / 'permutation_nulls.csv').D4_visit_geometry_r, d['visit_stat'].pearson_r, d['visit_stat'].permutation_p)
            require(d['visit_stat'].visit_transitions == expected.sum(), 'Visit edge count differs.')
            close(d['visit_stat'][['event_adjacency_r', 'event_adjacency_p']], d['event_stat'][['pearson_r', 'permutation_p']], 'Visit event-reference statistic differs.')
            full = row(four, assignment_id=aid, participant_omitted='none', transition_unit='basin_visit', geometry_D=4)
            close(full[['pearson_r', 'permutation_p']], d['visit_stat'][['pearson_r', 'permutation_p']], 'Visit aggregate differs.')
        data[aid] = d
        audit.append({'assignment_id': aid, 'events': len(ev), 'transitions': len(edges), 'frozen_memberships_match': True, 'profile_tests_checked': 4, 'order_tests_checked': 4, 'selected_geometry_and_visit_checked': aid in SELECTED})
    return (data, profiles, orders, manifest, audit)

def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=0.6)

def common_bounds(arrays, pad=0.16):
    values = np.concatenate(arrays)
    low = values.min(0)
    high = values.max(0)
    span = np.maximum(high - low, 0.1)
    return np.column_stack((low - pad * span, high + pad * span))
OMISSIONS = ['none', 'p01', 'p02', 'p03', 'p04', 'p05']
BASIN_MARKER_AREA = 55
EVENT_COLOR = '#285E83'

def title(ax, text):
    ax.set_title(text, loc='left', fontsize=11, weight='semibold', pad=12)

def draw_map(ax, d, kind, bounds, heading):
    if kind == 'mca':
        xy = d['centroids'][['Dim0', 'Dim3']].to_numpy()
        names = ['Dim0', 'Dim3']
    else:
        xy = d['pca'].iloc[:, :2].to_numpy()
        names = ['Successor PC1', 'Successor PC2']
    if kind == 'mca':
        ax.scatter(*xy.T, s=BASIN_MARKER_AREA, c=PALETTE[:d['k']], edgecolors='black', linewidth=0.65, zorder=3)
    else:
        ax.scatter(*xy.T, s=BASIN_MARKER_AREA, facecolors='white', edgecolors=PALETTE[:d['k']], linewidth=1.25, zorder=3)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    tidy(ax)
    title(ax, heading)
    return (ax, xy, d['labels'])

def draw_distance(ax, d, limits):
    pairs = d['pairs']
    x = pairs.geometry_distance.to_numpy()
    y = pairs.predictive_distance.to_numpy()
    ax.scatter(x, y, s=29, color=EVENT_COLOR, alpha=0.85, edgecolors='white', linewidth=0.35, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, intercept + slope * xx, color='#545C64', lw=1.1, zorder=2)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xlabel('4D centroid distance')
    ax.set_ylabel('Complete successor-profile distance')
    tidy(ax)
    title(ax, 'Full-space relationship')

def draw_alignment(ax, d, bounds):
    saved = d['aligned']
    geometry = saved[['terrain_x', 'terrain_y']].to_numpy()
    successor = saved[['predictive_x', 'predictive_y']].to_numpy()
    for i, color in enumerate(PALETTE[:d['k']]):
        ax.plot([geometry[i, 0], successor[i, 0]], [geometry[i, 1], successor[i, 1]], color='#B1B6BC', lw=0.75, zorder=1)
        ax.scatter(*geometry[i], s=BASIN_MARKER_AREA, c=color, edgecolors='black', linewidth=0.65, zorder=3)
        ax.scatter(*successor[i], s=BASIN_MARKER_AREA, facecolors='white', edgecolors=color, linewidth=1.25, zorder=4)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Normalized aligned coordinate 1')
    ax.set_ylabel('Normalized aligned coordinate 2')
    tidy(ax)
    title(ax, '2-D Procrustes alignment')
    return (ax, np.vstack([geometry, successor]), d['labels'] + d['labels'])
