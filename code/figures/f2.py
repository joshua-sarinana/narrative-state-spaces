"""Deterministic analysis of the frozen 256-event dataset."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import itertools
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone
PUBLIC_ROOT = Path(__file__).resolve().parents[2]
HERE = PUBLIC_ROOT / 'outputs/figure_sources/f2'
REFIT_ROOT = PUBLIC_ROOT
FLOW = dict(bins=25, pad_frac=0.05, density_sigma=1.0, flow_sigma=0.75, min_flow_count=0.05, contour_levels=18, density_max=2.1, stream_density=1.2, gaussian_boundary_mode='nearest')
SIZES = {1: 133, 2: 59, 3: 17, 4: 47}
PARTICIPANT_EVENTS = {'p01': 29, 'p02': 56, 'p03': 66, 'p04': 34, 'p05': 71}
COLORS = {1: '#17becf', 2: '#e377c2', 3: '#d62728', 4: '#1f77b4'}
DESCRIPTIONS = {1: 'Social context integration', 2: 'Goal-directed/prospective organization', 3: 'Virtual/digital mediation', 4: 'Physical/spatial organization'}
PROFILE_ANCHORS = {1: [('context_type_primary', 'social'), ('primary_hipp_mech', 'Context integration')], 2: [('primary_hipp_mech', 'Goal-directed navigation'), ('orient', 'prospective')], 3: [('context_type_primary', 'virtual_digital'), ('landmark', 'digital_space')], 4: [('context_type_primary', 'physical'), ('primary_hipp_mech', 'Place coding')]}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_and_validate():
    require((REFIT_ROOT/'outputs/mca').is_dir(),'Missing computed MCA results.')
    require((HERE / 'sources.json').is_file(), 'Missing package file: sources.json')
    manifest = json.loads((HERE / 'sources.json').read_text())
    tables = {}
    for role, entry in manifest['files'].items():
        path = REFIT_ROOT / entry['path_from_0_REFIT']
        require(path.is_file(), f'Missing current analysis input: {path}')
        require(sha256(path) == entry['sha256'], f'Analysis input differs from reviewed version: {path.name}')
        tables[role] = pd.read_csv(path, keep_default_na=False)
    events, edges, pairs = (tables[k] for k in ('events', 'edges', 'pairs'))
    key = ['participant_id_norm', 'hippo_id_norm']
    require(len(events) == 256 and (not events.duplicated(key).any()), 'Expected 256 unique event keys.')
    require(events.groupby('participant_id_norm').size().to_dict() == PARTICIPANT_EVENTS, 'Participant event counts differ from the frozen REFIT data.')
    require(events.groupby('cluster_k4_d4').size().to_dict() == SIZES, 'Four-cluster membership counts differ from the frozen REFIT data.')
    profiles = tables['cluster_profiles'].query("model_id == 'D4_K4'")
    require(len(profiles) == 4 and (not profiles.cluster.duplicated().any()) and (profiles.set_index('cluster').n_events.to_dict() == SIZES), 'Macrocluster profile IDs or counts disagree with the plotted memberships.')
    cells = tables['cluster_feature_cells'].query("model_id == 'D4_K4'")
    for cluster, anchors in PROFILE_ANCHORS.items():
        for variable, level in anchors:
            match = cells.loc[(cells.cluster == cluster) & (cells.variable == variable) & (cells.level == level)]
            require(len(match) == 1 and float(match.iloc[0].adjusted_standardized_residual) > 0, f'Cluster description lacks its reviewed source cell: C{cluster}, {variable}={level}')
    require(np.isfinite(events[[f'Dim{i}' for i in range(4)]].to_numpy()).all(), 'Nonfinite 4D coordinates.')
    require(len(edges) == 251 and (not edges.duplicated(['participant_id_norm', 'source_event']).any()), 'Expected 251 distinct source-event links.')
    lookup = events.set_index(key)
    counts = np.zeros((4, 4), dtype=int)
    records = []
    for row in edges.itertuples(index=False):
        src = lookup.loc[row.participant_id_norm, row.source_event]
        dst = lookup.loc[row.participant_id_norm, row.target_event]
        require(int(src.cluster_k4_d4) == row.source_cluster and int(dst.cluster_k4_d4) == row.target_cluster, 'Edge and event cluster labels disagree.')
        counts[row.source_cluster - 1, row.target_cluster - 1] += 1
        rec = row._asdict()
        for d in range(4):
            rec[f'source_Dim{d}'] = float(src[f'Dim{d}'])
            rec[f'target_Dim{d}'] = float(dst[f'Dim{d}'])
            rec[f'delta_Dim{d}'] = float(dst[f'Dim{d}'] - src[f'Dim{d}'])
        records.append(rec)
    links = pd.DataFrame(records)
    for participant, group in edges.groupby('participant_id_norm'):
        ids = set(events.loc[events.participant_id_norm == participant, 'hippo_id_norm'])
        targets = list(group.target_event)
        require(len(targets) == len(set(targets)), 'Multiple predecessors in a participant chain.')
        successors = dict(zip(group.source_event, group.target_event))
        starts = ids - set(targets)
        require(len(starts) == 1 and len(group) == len(ids) - 1, 'Invalid participant chain size.')
        visited = set()
        cursor = next(iter(starts))
        while cursor not in visited:
            visited.add(cursor)
            if cursor not in successors:
                break
            cursor = successors[cursor]
        require(visited == ids and cursor not in successors, 'Disconnected or cyclic participant chain.')
    saved_counts = tables['counts'].iloc[:, 1:].to_numpy(int)
    require(tables['counts'].iloc[:, 0].astype(str).tolist() == ['C1', 'C2', 'C3', 'C4'] and tables['probabilities'].iloc[:, 0].astype(str).tolist() == ['C1', 'C2', 'C3', 'C4'], 'Macro matrix row order differs from C1-C4.')
    require(list(tables['counts'].columns[1:]) == ['C1', 'C2', 'C3', 'C4'] and list(tables['probabilities'].columns[1:]) == ['C1', 'C2', 'C3', 'C4'], 'Macro matrix column order differs from C1-C4.')
    require(np.array_equal(counts, saved_counts), 'Reconstructed macro transition counts differ.')
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    require(np.allclose(probabilities, tables['probabilities'].iloc[:, 1:].to_numpy(float), rtol=0, atol=1e-12), 'Macro transition probabilities differ.')
    require(len(pairs) == 12 and (not pairs.duplicated(['source_cluster', 'target_cluster']).any()), 'Expected 12 unique observed non-self directed pairs.')
    centroids = events.groupby('cluster_k4_d4')[[f'Dim{i}' for i in range(4)]].mean()
    for row in pairs.itertuples(index=False):
        i, j = (row.source_cluster - 1, row.target_cluster - 1)
        require(i != j and row.transition_count == counts[i, j] and (counts[i, j] > 0), 'Plotted pair count does not match the event links.')
        require(np.isclose(row.transition_probability, probabilities[i, j], atol=1e-12, rtol=0), 'Plotted probability does not retain the full row denominator.')
        distance = np.linalg.norm(centroids.loc[i + 1] - centroids.loc[j + 1])
        require(np.isclose(row.centroid_distance_D4, distance, atol=1e-12, rtol=0), 'Plotted distance does not match the full 4D centroid distance.')
    summary = tables['summary'].query('mca_dimensions == 4').iloc[0]
    r = float(np.corrcoef(pairs.centroid_distance_D4, pairs.transition_probability)[0, 1])
    require(np.isclose(r, summary.pearson_r, atol=1e-12, rtol=0), 'Panel C Pearson r differs.')
    permutations = tables['permutations'].query('mca_dimensions == 4')
    require(len(permutations) == permutations.permutation.nunique() == 24, 'Expected all 24 distinct centroid-label permutations.')
    identity = permutations.loc[permutations.is_identity.astype(str).str.lower() == 'true']
    require(len(identity) == 1 and np.isclose(identity.pearson_r.iloc[0], r, atol=1e-12, rtol=0), 'The saved identity permutation differs from the observed statistic.')
    p = float(np.mean(np.abs(permutations.pearson_r.to_numpy()) >= abs(r) - 1e-12))
    require(np.isclose(p, summary.exact_two_sided_p, atol=1e-12, rtol=0), 'Panel C exact p differs.')
    centers = centroids.loc[[1, 2, 3, 4]].to_numpy(float)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    mask = (counts > 0) & ~np.eye(4, dtype=bool)
    reconstructed_null = np.array([np.corrcoef(distances[np.ix_(order, order)][mask], probabilities[mask])[0, 1] for order in itertools.permutations(range(4))])
    require(np.allclose(np.sort(reconstructed_null), np.sort(permutations.pearson_r), rtol=0, atol=1e-12), 'Reconstructed 24-label null differs from stored results.')
    projections = tables['projections'].query('reference_dimensions == 4').copy()
    require(len(projections) == 6, 'Expected six 4D axis-pair projection candidates.')
    best = projections.sort_values('descriptive_rank').iloc[0]
    require(best.projection == 'Dim0xDim3', 'The frozen 4D projection winner changed.')
    return (events, links, pairs, summary, projections, manifest)

def grid_and_flow(events, links):
    xy = events[['Dim0', 'Dim3']].to_numpy(float)
    lo, hi = (xy.min(axis=0), xy.max(axis=0))
    pad = FLOW['pad_frac'] * (hi - lo)
    xe = np.linspace(lo[0] - pad[0], hi[0] + pad[0], FLOW['bins'] + 1)
    ye = np.linspace(lo[1] - pad[1], hi[1] + pad[1], FLOW['bins'] + 1)
    xc, yc = ((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2)

    def hist(x, y, weights=None):
        return np.histogram2d(x, y, bins=(xe, ye), weights=weights)[0].T
    density_raw = hist(xy[:, 0], xy[:, 1])
    background = np.log1p(gaussian_filter(density_raw, FLOW['density_sigma'], mode='nearest'))
    counts = hist(links.source_Dim0, links.source_Dim3)
    smoothed = gaussian_filter(counts, FLOW['flow_sigma'], mode='nearest')
    velocities = []
    for dim in (0, 3):
        sums = hist(links.source_Dim0, links.source_Dim3, links[f'delta_Dim{dim}'])
        numerator = gaussian_filter(sums, FLOW['flow_sigma'], mode='nearest')
        velocities.append(np.divide(numerator, smoothed, out=np.zeros_like(smoothed), where=smoothed > 0))
    require(density_raw.sum() == 256 and counts.sum() == 251, 'Histogram dropped an event or source link.')
    require(background.max() <= FLOW['density_max'], 'Density exceeds the declared color scale.')
    require((xy[:, 0] >= xc[0]).all() and (xy[:, 0] <= xc[-1]).all() and (xy[:, 1] >= yc[0]).all() and (xy[:, 1] <= yc[-1]).all(), 'Center-based limits clip events.')
    X, Y = np.meshgrid(xc, yc)
    grid = pd.DataFrame({'Dim0': X.ravel(), 'Dim3': Y.ravel(), 'event_count': density_raw.ravel(), 'log1p_smoothed_event_count': background.ravel(), 'source_transition_count': counts.ravel(), 'smoothed_source_count': smoothed.ravel(), 'mean_delta_Dim0': velocities[0].ravel(), 'mean_delta_Dim3': velocities[1].ravel(), 'flow_visible': (smoothed >= FLOW['min_flow_count']).ravel()})
    return (xc, yc, background, velocities, smoothed, grid)

def style_axes(ax, title, x, y, letter=None):
    ax.set_title(title, loc='left', fontsize=10.5, pad=11, fontweight='semibold')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.65)
    if letter:
        ax.text(-0.14, 1.055, letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

def draw_geometry(ax, events, axes=(0, 3), letter='A', limits=None):
    cols = [f'Dim{i}' for i in axes]
    for cluster, group in events.groupby('cluster_k4_d4', sort=True):
        xy = group[cols].to_numpy(float)
        color = COLORS[int(cluster)]
        unique = np.unique(xy, axis=0)
        if len(unique) >= 3 and np.linalg.matrix_rank(unique - unique.mean(axis=0)) == 2:
            hull = unique[ConvexHull(unique).vertices]
            ax.add_patch(Polygon(hull, fill=False, edgecolor=color, linewidth=0.85, alpha=0.58))
        ax.scatter(xy[:, 0], xy[:, 1], s=14, c=color, alpha=0.82, edgecolors='none', label=f'C{cluster}  n = {len(group)}', zorder=3)
        center = xy.mean(axis=0)
        ax.scatter(*center, marker='X', c=color, edgecolors='#202020', linewidths=0.65, s=63, zorder=5)
        ax.annotate(f'C{cluster}', center, xytext=(6, 7), textcoords='offset points', fontsize=8.5, fontweight='semibold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.8), zorder=6)
    style_axes(ax, '2-D Projection', cols[0], cols[1], letter)
    ax.set_aspect('equal', adjustable='box')
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    else:
        ax.margins(0.09)

def density_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('event_density', ['#fafafa', '#dde8e8', '#a4c1c6', '#668f9c'])

def draw_flow(ax, field, letter='B'):
    xc, yc, background, velocities, smoothed, _ = field
    levels = np.linspace(0, FLOW['density_max'], FLOW['contour_levels'] + 1)
    ax.contourf(xc, yc, background, levels=levels, cmap=density_colormap(), vmin=0, vmax=FLOW['density_max'])
    ax.contour(xc, yc, background, levels=np.linspace(0, FLOW['density_max'], 7)[1:], colors='#527580', linewidths=0.35, alpha=0.35)
    mask = smoothed < FLOW['min_flow_count']
    u, v = [np.ma.array(a, mask=mask) for a in velocities]
    ax.streamplot(xc, yc, u, v, density=FLOW['stream_density'], color='#172d35', linewidth=0.7, arrowsize=0.85)
    ax.set_xlim(xc[0], xc[-1])
    ax.set_ylim(yc[0], yc[-1])
    ax.set_aspect('equal', adjustable='box')
    style_axes(ax, 'Global Single-Step Transitions', 'Dim0', 'Dim3', letter)

def draw_distance(ax, pairs, letter='C'):
    x, y = (pairs.centroid_distance_D4.to_numpy(), pairs.transition_probability.to_numpy())
    ax.scatter(x, y, s=29, color='#777777', edgecolors='#222222', linewidths=0.35, zorder=3)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 150)
    ax.plot(xx, slope * xx + intercept, color='#456776', linewidth=1.15)
    style_axes(ax, 'Distance and Transition Probability', '4-D Cluster-Centroid Distance', 'Transition Probability', letter)
    ax.set_ylim(0, max(y) * 1.2)
    ax.margins(x=0.07)
