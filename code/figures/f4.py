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
HERE = PUBLIC_ROOT / 'outputs/figure_sources/f4'
ASSIGNMENTS = ['D4_K10_L9_U47', 'D2_K9_L8_U50']
LABELS = {ASSIGNMENTS[0]: '4-D 10-Basin', ASSIGNMENTS[1]: '2-D 9-Basin'}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
FAMILIES = ['context', 'mechanism', 'meta_theme']
FAMILY_NAMES = ['Context', 'Mechanism', 'Narrative theme']
CONTEXTS = ['internal_simulated', 'physical', 'social', 'virtual_digital']
CONTEXT_NAMES = ['Internal / simulated', 'Physical', 'Social', 'Virtual / digital']
KEYS = ['participant_id_norm', 'hippo_id_norm']
NODE_AREA = 64
OCCUPANCY_AREA_PER_EVENT = 4.0
ELEVATION, AZIMUTH = (28, 60)
EDGE_WIDTH = 0.8
PANEL_SIZE = (7.2, 5.6)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def close(a, b, message):
    require(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-12, rtol=0, equal_nan=True), message)

def normalize(counts):
    counts = np.asarray(counts, float)
    den = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, den, out=np.zeros_like(counts), where=den > 0)

def read_csv(path, matrix=False):
    return pd.read_csv(path, index_col=0 if matrix else None, keep_default_na=matrix)

def count_edges(edges, k):
    counts = np.zeros((k, k), int)
    np.add.at(counts, (edges.source_state.to_numpy(int) - 1, edges.target_state.to_numpy(int) - 1), 1)
    return counts

def selected_connections(counts):
    counts = np.asarray(counts, int)
    chosen = np.diag(np.diag(counts) > 0)
    for i, row in enumerate(counts):
        off = row.copy()
        off[i] = 0
        if off.max() > 0:
            chosen[i, off == off.max()] = True
    return chosen

def validate_and_load():
    manifest = json.loads((HERE / 'inputs/provenance.json').read_text())
    root = PUBLIC_ROOT
    canonical_checked = []
    for rel, meta in manifest['files'].items():
        path = HERE / 'inputs' / rel
        require(path.is_file(), f'Missing input: {rel}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256'], f'Changed input: {rel}')
        source = meta['source'].get('source_path', '')
        marker = '4_REFIT_Downstream/'
        if marker in source:
            canonical = root / (marker + source.split(marker, 1)[1])
            if canonical.is_file():
                require(hashlib.sha256(canonical.read_bytes()).hexdigest() == meta['sha256'], f'Canonical results differ from this frozen package: {canonical}. Refresh the package inputs before plotting.')
                canonical_checked.append(str(canonical.relative_to(root)))
    pooled = read_csv(HERE / 'inputs/conditioned_routing_all_assignments.csv')
    frozen = read_csv(HERE / 'inputs/frozen_memberships.csv')
    data = {}
    for aid in ASSIGNMENTS:
        folder = HERE / 'inputs' / aid
        events = read_csv(folder / 'inputs/event_memberships_features_coordinates.csv')
        edges = read_csv(folder / 'inputs/event_transition_records.csv')
        p = read_csv(folder / 'inputs/transition_probabilities_empirical.csv', True)
        counts = read_csv(folder / 'inputs/transition_counts.csv', True)
        labels = list(p.index)
        k = len(labels)
        require(labels == list(p.columns) == [f'B{i + 1}' for i in range(k)], 'Unexpected basin order.')
        require(len(events) == 256 and len(edges) == 251 and (events.participant_id_norm.nunique() == 5), 'Expected 256 events, 251 adjacencies, and five participants.')
        joined = events[KEYS + ['state']].merge(frozen[KEYS + [aid]], on=KEYS, validate='one_to_one')
        require(len(joined) == 256 and np.array_equal(joined.state, joined[aid]), 'Frozen memberships changed.')
        lookup = events.set_index(KEYS)
        for end in ('source', 'target'):
            keys = pd.MultiIndex.from_frame(edges[['participant_id_norm', end + '_event']])
            require(keys.isin(lookup.index).all(), 'An edge does not resolve within its participant.')
            require(np.array_equal(lookup.loc[keys, 'state'], edges[end + '_state']), 'Edge assignment changed.')
        source_keys = pd.MultiIndex.from_frame(edges[['participant_id_norm', 'source_event']])
        require(np.array_equal(lookup.loc[source_keys, 'context_type_primary'], edges.context_type_primary), 'Conditioning must use the source-event context.')
        close(count_edges(edges, k), counts, 'Empirical transition counts differ.')
        close(normalize(counts), p, 'Empirical probabilities differ.')
        d = dict(aid=aid, events=events, edges=edges, k=k, labels=labels, centroids=events.groupby('state')[[f'Dim{i}' for i in range(4)]].mean().reindex(range(1, k + 1)), networks={'unconditional': dict(P=p.to_numpy(), counts=counts.to_numpy(int), support=counts.sum(axis=1).to_numpy(int))})
        d['occupancy'] = events.groupby('state').size().reindex(range(1, k + 1)).to_numpy(int)
        require(int(d['occupancy'].sum()) == 256, 'Basin occupancies must sum to 256.')
        context_root = folder / 'conditioned_context'
        summary = read_csv(context_root / 'condition_summary.csv')
        require(set(summary.condition) == set(CONTEXTS) and summary.n_transitions.sum() == 251, 'Context groups must partition the 251 adjacencies.')
        for row in summary.itertuples():
            cp = read_csv(context_root / f'{row.condition_id}_routing_matrix.csv', True)
            support = read_csv(context_root / f'{row.condition_id}_row_support.csv').set_index('basin')
            subset = edges[edges.context_type_primary == row.condition]
            cc = count_edges(subset, k)
            require(list(cp.index) == list(cp.columns) == labels, 'Context matrix order differs.')
            close(cc.sum(axis=1), support.loc[labels, 'outgoing_transitions'], 'Context support differs.')
            close(normalize(cc), cp, 'Context probabilities differ.')
            d['networks'][row.condition] = dict(P=cp.to_numpy(), counts=cc, support=cc.sum(axis=1))
        sim = read_csv(context_root / 'full_matrix_cosine_similarity.csv', True).loc[CONTEXTS, CONTEXTS]
        flat = np.asarray([d['networks'][c]['P'].ravel() for c in CONTEXTS])
        unit = flat / np.linalg.norm(flat, axis=1, keepdims=True)
        close(unit @ unit.T, sim, 'Between-context similarity does not match the complete saved matrices.')
        d['similarity'] = sim
        stats = pooled[(pooled.assignment_id == aid) & (pooled.test_family == 'full_matrix')].set_index('field_key').loc[FAMILIES].copy()
        require(len(stats) == 3 and (stats.n_perm == 10000).all() and (stats.seed == 42).all(), 'Unexpected inference settings.')
        require((stats.min_condition_transitions == 10).all(), 'Unexpected minimum condition support.')
        pvals = stats.lower_tail_p_value.to_numpy(float)
        order = np.argsort(pvals)
        adjusted = np.empty(3)
        adjusted[order] = np.minimum.accumulate((pvals[order] * 3 / np.arange(1, 4))[::-1])[::-1].clip(max=1)
        close(adjusted, stats.bh_q_within_assignment_and_test_family, 'Local BH adjustment differs.')
        close(np.mean(sim.to_numpy()[np.triu_indices(4, 1)]), stats.loc['context', 'observed_stat'], 'Context summary differs.')
        stats['contrast'] = stats.null_mean - stats.observed_stat
        d['stats'] = stats
        evxy = events[['Dim0', 'Dim3']].to_numpy()
        lo, hi = (evxy.min(axis=0), evxy.max(axis=0))
        pad = (0.05 - 1.1 / 50) * (hi - lo)
        d['limits'] = (lo - pad, hi + pad)
        data[aid] = d
    require(data[ASSIGNMENTS[0]]['events'].drop(columns='state').equals(data[ASSIGNMENTS[1]]['events'].drop(columns='state')), 'Event geometry or annotations differ across assignments.')
    return (data, manifest, canonical_checked)

def inches_axes(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])

def panel_title(fig, letter, title):
    fig.text(0.035, 0.945, title, fontsize=12, va='center')

def save_png(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, facecolor='white')
    atomic_png(buffer.getvalue(), path)
    plt.close(fig)

def atomic_png(payload, path):
    with Image.open(io.BytesIO(payload)) as check:
        check.verify()
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def draw_context_similarity(d, path, dpi):
    fig = plt.figure(figsize=PANEL_SIZE)
    ax = inches_axes(fig, 1.45, 1.1, 4.25, 4.05)
    values = d['similarity'].to_numpy()
    close(np.diag(values), np.ones(len(CONTEXTS)), 'Context self-similarity must be one.')
    close(values, values.T, 'Context cosine similarity must be symmetric.')
    names = CONTEXT_NAMES
    cmap = plt.get_cmap('viridis').copy()
    im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, interpolation='nearest', aspect='equal')
    ax.set_xticks(range(len(names)), names, rotation=32, ha='right', fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.tick_params(length=0, pad=6)
    cax = inches_axes(fig, 5.98, 1.18, 0.18, 3.55)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label('Cosine similarity', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    panel_title(fig, 'B', 'Between-context routing similarity')
    save_png(fig, path, dpi)

def bezier_points(vertices, n=101):
    t = np.linspace(0, 1, n)[:, None]
    a, b, c, d = vertices
    return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d

def rectangle_distance(points, rect):
    low, high = (rect[:2], rect[2:])
    delta = np.maximum(np.maximum(low - points, points - high), 0)
    return np.linalg.norm(delta, axis=1)

def place_labels(ax, centers, active, dense=False, radii=None):
    scale = ax.figure.dpi / 72
    renderer = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent()
    bounds = np.array([box.x0, box.y0, box.x1, box.y1]) / scale
    rectangles = []
    if radii is None:
        radii = np.full(len(centers), 4.0)
    for i, center in enumerate(centers):
        outward = center - np.median(centers, axis=0)
        outward = outward / max(np.linalg.norm(outward), 1e-09)
        text = f'B{i + 1}'
        width, height, _ = renderer.get_text_width_height_descent(text, FontProperties(size=9), False)
        size = np.array([width / scale + 3, height / scale + 4])
        choices = []
        for radius in np.asarray((16, 22, 29, 36) if dense else (11, 16, 22, 29)) + radii[i] - 4:
            for angle in np.arange(0, 360, 45):
                offset = radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                mid = center + offset
                rect = np.r_[mid - size / 2, mid + size / 2]
                collisions = sum((float(rectangle_distance(c[None], rect)[0] < radii[j] + 4) for j, c in enumerate(centers)))
                overlap = sum((max(0, min(rect[2], r[2]) - max(rect[0], r[0])) * max(0, min(rect[3], r[3]) - max(rect[1], r[1])) for r in rectangles))
                outside = np.maximum(bounds[:2] - rect[:2], 0).sum() + np.maximum(rect[2:] - bounds[2:], 0).sum()
                angular_cost = 180 * (1 - np.dot(offset / radius, outward))
                choices.append((1000000.0 * collisions + 10000.0 * overlap + 1000000.0 * outside + radius ** 2 + angular_cost, offset, rect))
        _, offset, rect = min(choices, key=lambda item: item[0])
        xy = ax.transData.inverted().transform(center * scale)
        ax.annotate(text, xy, xytext=offset, textcoords='offset points', ha='center', va='center', fontsize=9, color='#111111' if i in active else '#8E9398', zorder=9, bbox=dict(boxstyle='round,pad=.13', fc='white', ec='none', alpha=0.93) if dense else None)
        rectangles.append(rect)
    return (rectangles, bounds)

def route_cost(vertices, centers, boxes, bounds, i, j, existing, radii=None):
    points = bezier_points(vertices)
    if radii is None:
        radii = np.full(len(centers), 4.0)
    clearance = min((np.linalg.norm(points - c, axis=1).min() - radii[index] + 4 for index, c in enumerate(centers) if index not in (i, j)), default=1000.0)
    label_clearance = min((rectangle_distance(points, b).min() for b in boxes), default=1000.0)
    boundary = np.maximum(bounds[:2] + 3 - points, 0).sum() + np.maximum(points - (bounds[2:] - 3), 0).sum()
    collisions = max(0, 9 - clearance) ** 2 * 100000.0 + max(0, 3 - label_clearance) ** 2 * 100000.0
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    crowding = 0.0
    for old in existing:
        distances = np.linalg.norm(points[8:-8:4, None] - old[None, 8:-8:4], axis=2)
        crowding += np.exp(-np.min(distances, axis=1) / 3).sum()
    return (collisions + boundary * 100000.0 + length + crowding * 0.7, clearance, label_clearance)

def route_edge(centers, boxes, bounds, i, j, reciprocal, existing, radii=None):
    a, b = (centers[i], centers[j])
    variable_sizes = radii is not None
    if radii is None:
        radii = np.full(len(centers), 4.0)
    source_gap, target_gap = (radii[i] + 2, radii[j] + 4)
    candidates = []
    if i == j:
        for radius in np.asarray((16, 20, 24, 32, 42, 54)) + max(0, radii[i] - 4):
            for angle in np.arange(0, 360, 15):
                u = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                v = np.array([-u[1], u[0]])
                for shape in (0.45, 0.8, 1.2):
                    vertices = np.array([a + source_gap * u + 4 * v, a + radius * u + shape * radius * v, a + radius * u - shape * radius * v, a + target_gap * u - 4 * v])
                    score, nc, lc = route_cost(vertices, centers, boxes, bounds, i, j, existing, radii)
                    candidates.append((score, vertices, nc, lc))
    else:
        delta = b - a
        length = np.linalg.norm(delta)
        normal = np.array([-delta[1], delta[0]]) / length
        offsets = ([6, 12] if variable_sizes else []) + [18, 28, 42, 60, 85, 110] if reciprocal else [0, 12, -12, 22, -22, 36, -36, 55, -55, 80, -80, 110, -110]
        if length < 30:
            offsets = [value for value in offsets if abs(value) >= 18]
        for offset in offsets:
            c1, c2 = (a + delta / 3 + normal * offset, a + 2 * delta / 3 + normal * offset)
            start = a + source_gap * (c1 - a) / np.linalg.norm(c1 - a)
            end = b - target_gap * (b - c2) / np.linalg.norm(b - c2)
            vertices = np.array([start, c1, c2, end])
            score, nc, lc = route_cost(vertices, centers, boxes, bounds, i, j, existing, radii)
            candidates.append((score, vertices, nc, lc))
    best = min(candidates, key=lambda item: item[0])
    if i != j and (best[2] < 9 or best[3] < 3):
        angle = np.arctan2(delta[1], delta[0])
        for arm in (max(18, length / 3), max(28, length / 2)):
            for source_shift in range(-120, 121, 15):
                ua = np.array([np.cos(angle + np.deg2rad(source_shift)), np.sin(angle + np.deg2rad(source_shift))])
                for target_shift in range(-120, 121, 15):
                    end_angle = angle + np.pi + np.deg2rad(target_shift)
                    ub = np.array([np.cos(end_angle), np.sin(end_angle)])
                    vertices = np.array([a + source_gap * ua, a + (source_gap + arm) * ua, b + (target_gap + arm) * ub, b + target_gap * ub])
                    if reciprocal and np.dot(bezier_points(vertices, 3)[1] - (a + b) / 2, normal) < 3:
                        continue
                    score, nc, lc = route_cost(vertices, centers, boxes, bounds, i, j, existing, radii)
                    if score < best[0]:
                        best = (score, vertices, nc, lc)
    return best[1:]

def draw_network(d, which, path, dpi, letter, title):
    fig = plt.figure(figsize=PANEL_SIZE)
    ax = inches_axes(fig, 0.82, 0.65, 5.95, 4.3)
    low, high = d['limits']
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='Dim0', ylabel='Dim3', aspect='equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    xy = d['centroids'][['Dim0', 'Dim3']].to_numpy()
    ax.scatter(*xy.T, s=NODE_AREA, c=COLORS[:d['k']], edgecolors='black', linewidths=0.7, zorder=8)
    panel_title(fig, letter, title)
    fig.canvas.draw()
    scale = fig.dpi / 72
    centers = ax.transData.transform(xy) / scale
    boxes, bounds = place_labels(ax, centers, set(range(d['k'])))
    net = d['networks'][which]
    selected = selected_connections(net['counts'])
    chosen = list(zip(*np.nonzero(selected)))
    keys = set(chosen)
    chosen.sort(key=lambda ij: (ij[0] != ij[1], -np.linalg.norm(centers[ij[1]] - centers[ij[0]]), ij))
    existing, records = ([], [])
    for i, j in chosen:
        vertices, nc, lc = route_edge(centers, boxes, bounds, i, j, (j, i) in keys and i != j, existing)
        path_xy = ax.transData.inverted().transform(vertices * scale)
        curve = MplPath(path_xy, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        p = float(net['P'][i, j])
        width = EDGE_WIDTH
        low_support = net['support'][i] < 3
        arrow = FancyArrowPatch(path=curve, arrowstyle='-|>', mutation_scale=9, lw=width, color='#34495A', linestyle=(0, (2.5, 2.0)) if low_support else 'solid', zorder=4, capstyle='round', joinstyle='round')
        arrow.set_path_effects([patheffects.Stroke(linewidth=width + 1.1, foreground='white'), patheffects.Normal()])
        ax.add_patch(arrow)
        existing.append(bezier_points(vertices))
        records.append(dict(assignment=d['aid'], panel=which, source=d['labels'][i], destination=d['labels'][j], display_projection='Dim0_Dim3', count=int(net['counts'][i, j]), probability=p, source_outgoing=int(net['support'][i]), self_transition=i == j, linewidth_points=width, low_support=bool(low_support), nearest_other_node_points=float(nc), nearest_label_points=float(lc), **{f'control_{m}_{axis}': float(path_xy[m, q]) for m in range(4) for q, axis in enumerate(('x', 'y'))}))
    save_png(fig, path, dpi)
    return records

def empirical_arrow_location(vertices, centers, radii, boxes, arrow_points):
    points = bezier_points(vertices, 201)
    distances = np.linalg.norm(points[:, None] - centers[None], axis=2) - radii[None]
    node_clear = distances.min(axis=1)
    label_clear = np.min(np.asarray([rectangle_distance(points, box) for box in boxes]), axis=0)
    clearance = np.minimum(node_clear, label_clear)
    choices = []
    for tip in range(25, 181):
        stem = tip - 1
        while stem > 0 and np.linalg.norm(points[tip] - points[stem]) < 7:
            stem -= 1
        space = float(clearance[stem:tip + 1].min())
        crowd = min((np.linalg.norm(points[tip] - q) for q in arrow_points), default=1000.0)
        penalty = max(0, 4 - space) ** 2 * 100000.0 + max(0, 5 - crowd) ** 2 * 1000.0 + abs(tip - 120) * 0.025
        choices.append((penalty, tip, stem, space))
    return (*min(choices), points)

def draw_empirical_3d(d, depth, path, dpi):
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.ticker import MaxNLocator
    from matplotlib.patches import PathPatch
    fig = plt.figure(figsize=PANEL_SIZE)
    ax = fig.add_axes([0.035, 0.19, 0.9, 0.7], projection='3d')
    xyz = d['centroids'][['Dim0', depth, 'Dim3']].to_numpy()
    lo, hi = (xyz.min(axis=0), xyz.max(axis=0))
    pad = 0.13 * (hi - lo)
    ax.set(xlim=(hi[0] + pad[0], lo[0] - pad[0]), ylim=(lo[1] - pad[1], hi[1] + pad[1]), zlim=(lo[2] - pad[2], hi[2] + pad[2]))
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_proj_type('persp')
    ax.set_box_aspect((4, 4, 3), zoom=1.08)
    ax.set_xlabel('Dim0', labelpad=6, fontsize=10)
    ax.set_ylabel(depth, labelpad=6, fontsize=10)
    ax.set_zlabel('Dim3', labelpad=5, fontsize=10)
    ax.tick_params(labelsize=8, pad=1)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_pane_color((0.95, 0.95, 0.95, 1))
        axis._axinfo['grid'].update(color=(0.64, 0.66, 0.68, 0.75), linewidth=0.6, linestyle='-')
    panel_title(fig, 'A', 'Empirical Transition Network')
    fig.canvas.draw()
    projection = np.column_stack(proj3d.proj_transform(*xyz.T, ax.get_proj())[:2])
    pixel_xy = ax.transData.transform(projection)
    overlay = fig.add_axes([0, 0.165, 1, 0.755], frameon=False)
    overlay.set(xlim=(0, 1), ylim=(0, 1))
    overlay.axis('off')
    pos = overlay.transData.inverted().transform(pixel_xy)
    area = OCCUPANCY_AREA_PER_EVENT * d['occupancy']
    overlay.scatter(*pos.T, s=area, c=COLORS[:d['k']], edgecolors='black', linewidths=0.9, zorder=8)
    scale = fig.dpi / 72
    centers = pixel_xy / scale
    radii = np.sqrt(area) / 2 + 0.45
    boxes, bounds = place_labels(overlay, centers, set(range(d['k'])), dense=True, radii=radii)
    net = d['networks']['unconditional']
    chosen = list(zip(*np.nonzero(net['counts'] > 0)))
    keys = set(chosen)
    chosen.sort(key=lambda ij: (ij[0] != ij[1], -np.linalg.norm(centers[ij[1]] - centers[ij[0]]), ij))
    existing, records, arrow_points = ([], [], [])
    for i, j in chosen:
        delta = centers[j] - centers[i]
        length = np.linalg.norm(delta)
        if i == j or length < radii[i] + radii[j] + 18:
            vertices, nc, lc = route_edge(centers, boxes, bounds, i, j, i != j and (j, i) in keys, existing, radii)
        else:
            normal = np.array([-delta[1], delta[0]]) / length
            bend = 4.0 if (j, i) in keys else 0.0
            a, b = (centers[i], centers[j])
            c1, c2 = (a + delta / 3 + bend * normal, a + 2 * delta / 3 + bend * normal)
            start = a + (radii[i] + 2) * (c1 - a) / np.linalg.norm(c1 - a)
            end = b - (radii[j] + 4) * (b - c2) / np.linalg.norm(b - c2)
            vertices = np.array([start, c1, c2, end])
            _, nc, lc = route_cost(vertices, centers, boxes, bounds, i, j, [], radii)
        location = empirical_arrow_location(vertices, centers, radii, boxes, arrow_points)
        if location[3] < 3:
            alternatives = []
            if i == j:
                for growth in (1.2, 1.6, 2.1):
                    for angle in np.arange(0, 360, 30):
                        t = np.deg2rad(angle)
                        rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                        proposal = (vertices - centers[i]) @ rotation.T
                        proposal[1:3] *= growth
                        alternatives.append(proposal + centers[i])
            else:
                normal = np.array([-delta[1], delta[0]]) / length
                for bend in (8, -8, 16, -16, 24, -24, 36, -36, 54, -54):
                    a, b = (centers[i], centers[j])
                    c1, c2 = (a + delta / 3 + bend * normal, a + 2 * delta / 3 + bend * normal)
                    start = a + (radii[i] + 2) * (c1 - a) / np.linalg.norm(c1 - a)
                    end = b - (radii[j] + 4) * (b - c2) / np.linalg.norm(b - c2)
                    alternatives.append(np.array([start, c1, c2, end]))
            best = (location[0], vertices, location)
            for proposal in alternatives:
                candidate = empirical_arrow_location(proposal, centers, radii, boxes, arrow_points)
                if candidate[0] < best[0]:
                    best = (candidate[0], proposal, candidate)
                if candidate[3] >= 4:
                    break
            _, vertices, location = best
        _, tip, stem, arrow_clear, points = location
        require(arrow_clear >= 3, f"Direction marker obscured: {d['aid']} {depth} B{i + 1} to B{j + 1}.")
        _, nc, lc = route_cost(vertices, centers, boxes, bounds, i, j, [], radii)
        path_xy = overlay.transData.inverted().transform(vertices * scale)
        curve = MplPath(path_xy, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        line = PathPatch(curve, fill=False, lw=EDGE_WIDTH, color='#48799D', alpha=0.48, zorder=4, capstyle='round', joinstyle='round')
        overlay.add_patch(line)
        arrow_xy = overlay.transData.inverted().transform(points[[stem, tip]] * scale)
        arrow = FancyArrowPatch(arrow_xy[0], arrow_xy[1], arrowstyle='-|>', mutation_scale=7, shrinkA=0, shrinkB=0, lw=EDGE_WIDTH, color='#48799D', alpha=0.85, zorder=5, capstyle='round', joinstyle='round')
        overlay.add_patch(arrow)
        arrow_points.append(points[tip])
        existing.append(points[::2])
        records.append(dict(assignment=d['aid'], panel='unconditional', display_projection=f'Dim0_{depth}_Dim3', source=d['labels'][i], destination=d['labels'][j], count=int(net['counts'][i, j]), probability=float(net['P'][i, j]), source_outgoing=int(net['support'][i]), self_transition=i == j, linewidth_points=EDGE_WIDTH, low_support=False, nearest_other_node_adjusted_points=float(nc), nearest_label_points=float(lc), direction_marker_clearance_points=arrow_clear, source_node_radius_points=float(radii[i]), target_node_radius_points=float(radii[j]), source_clearance_points=float(np.linalg.norm(vertices[0] - centers[i]) - radii[i]), target_clearance_points=float(np.linalg.norm(vertices[-1] - centers[j]) - radii[j]), **{f'control_{m}_{axis}': float(path_xy[m, q]) for m in range(4) for q, axis in enumerate(('x', 'y'))}))
    require(len(records) == int((net['counts'] > 0).sum()), 'Missing observed empirical connections.')
    require(sum((r['count'] for r in records)) == 251, 'Empirical display must account for every adjacency.')
    probe = xyz.mean(axis=0)
    pp = np.column_stack(proj3d.proj_transform(*np.array([probe, probe + [1, 0, 0], probe + [0, 0, 1]]).T, ax.get_proj())[:2])
    require(pp[1, 0] > pp[0, 0] and pp[2, 1] > pp[0, 1], 'Dim0 must increase rightward and Dim3 upward.')
    draw_size_key_inside(fig)
    save_png(fig, path, dpi)
    return records

def draw_contrast(d, path, dpi):
    fig = plt.figure(figsize=(4.8, 5.6))
    ax = inches_axes(fig, 1.44, 0.92, 3.08, 3.85)
    vals = d['stats'].contrast.to_numpy(float)
    ax.axvline(0, color='#909090', lw=0.8)
    ax.hlines([2, 1, 0], 0, vals, color='#798FA1', lw=2)
    ax.scatter(vals, [2, 1, 0], color='#34495A', s=45, zorder=3)
    ax.set(xlim=(-0.008, 0.2), ylim=(-0.5, 2.5), yticks=[2, 1, 0], yticklabels=FAMILY_NAMES, xticks=[0, 0.05, 0.1, 0.15, 0.2], xlabel='Shuffled mean − observed\nmean cosine similarity')
    ax.tick_params(axis='y', length=0, labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    panel_title(fig, 'E', 'Full-network conditioning contrast')
    save_png(fig, path, dpi)

def draw_size_key_inside(fig):
    ax = fig.add_axes([0.12, 0.018, 0.76, 0.125], zorder=20)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.text(0.5, 0.73, 'Basin event count (circle area)', ha='center', va='center', fontsize=9)
    for x, count in zip([0.16, 0.43, 0.7], [10, 25, 50]):
        ax.scatter([x], [0.3], s=OCCUPANCY_AREA_PER_EVENT * count, c='#AAAAAA', edgecolors='black', linewidths=0.9)
        ax.text(x + 0.05, 0.3, f'{count} events', fontsize=8.5, va='center')
