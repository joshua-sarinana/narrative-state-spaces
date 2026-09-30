"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import PercentFormatter
AIDS = ['D4_K10_L9_U47', 'D4_K9_L8_U49', 'D4_K6_L8_U50', 'D2_K9_L8_U50', 'D2_K8_L8_U39']
MAIN = [AIDS[0], AIDS[3]]
LABELS = {a: f"{a.split('_')[0][1:]}-D {a.split('_')[1][1:]}-Basin" for a in AIDS}
BLUE = '#22689E'
ORANGE = '#B66A1D'
INK = '#263238'
GRAY = '#B9C3CA'
COLORS = {a: BLUE if a.startswith('D4') else ORANGE for a in AIDS}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
PANEL_FILES = {'S1A': 'S1A_MCA_parallel_analysis.png', 'S1C': 'S1C_MCA_inertia.png', 'S1B': 'S1B_MCA_full_space_fidelity.png', 'S1D': 'S1D_MCA_omission_retention.png', 'S2A': 'S2A_Context_social.png', 'S2B': 'S2B_Context_physical.png', 'S2C': 'S2C_Context_internal_simulated.png', 'S2D': 'S2D_Context_virtual_digital.png', 'S2E': 'S2E_Anchor_social.png', 'S2F': 'S2F_Anchor_static_landmark.png', 'S2G': 'S2G_Anchor_goal.png', 'S2H': 'S2H_Anchor_dynamic_landmark.png', 'S3A': 'S3A_Geometric_tradeoffs.png', 'S3B': 'S3B_Basin_occupancy.png', 'S3C': 'S3C_Basin_mean_silhouette.png', 'S4B': 'S4B_All11_feature_routing_4D_10B.png', 'S4A': 'S4A_All11_feature_routing_2D_9B.png', 'S5A': 'S5A_2D_9B_Representational_projection.png', 'S5B': 'S5B_2D_9B_Successor_PCA.png', 'S5C': 'S5C_2D_9B_Procrustes_alignment.png', 'S5D': 'S5D_2D_9B_Complete_profile_distance.png', 'S6A': 'S6A_4D_10B_Visit_collapse.png', 'S6B': 'S6B_2D_9B_Visit_collapse.png', 'S6C': 'S6C_Profile_definition_controls.png', 'S6D': 'S6D_Narrative_order_controls.png'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def close(a, b, message, tol=1e-12):
    require(np.allclose(a, b, atol=tol, rtol=0, equal_nan=True), message)

def truth(s):
    return s.astype(str).str.lower().eq('true')

def ptext(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=0.65)

def title(ax, s):
    ax.set_title(s, loc='left', fontsize=11, weight='semibold', pad=13)

def row(d, **values):
    for k, v in values.items():
        d = d[d[k] == v]
    require(len(d) == 1, f'Expected unique row {values}')
    return d.iloc[0]

def bounds(arrays, pad=0.18):
    a = np.concatenate(arrays)
    lo = a.min(0)
    hi = a.max(0)
    span = np.maximum(hi - lo, 0.1)
    return np.column_stack((lo - pad * span, hi + pad * span))

def basin_labels(ax, xy, labels):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    factor = ax.figure.dpi / 72
    centers = ax.transData.transform(xy)
    boxlim = ax.get_window_extent()
    placed = []

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    for label, center, pixel in zip(labels, xy, centers):
        w, h, _ = renderer.get_text_width_height_descent(label, FontProperties(size=8), False)
        w += 4 * factor
        h += 5 * factor
        options = []
        for r in [10, 17, 25, 34]:
            for dx, dy in [(r, r), (r, -r), (-r, r), (-r, -r), (r, 0), (-r, 0), (0, r), (0, -r)]:
                pos = pixel + np.array([dx, dy]) * factor
                box = (pos[0] - w / 2, pos[1] - h / 2, pos[0] + w / 2, pos[1] + h / 2)
                outside = max(0, boxlim.x0 - box[0]) + max(0, boxlim.y0 - box[1]) + max(0, box[2] - boxlim.x1) + max(0, box[3] - boxlim.y1)
                score = 10000 * outside + 100 * sum((overlap(box, b) for b in placed)) + 40 * sum((overlap(box, (p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5)) for p in centers)) + dx * dx + dy * dy
                options.append((score, dx, dy, box))
        _, dx, dy, box = min(options, key=lambda v: v[0])
        placed.append(box)
        ax.annotate(label, center, xytext=(dx, dy), textcoords='offset points', ha='center', va='center', fontsize=8, zorder=8, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.86, 'pad': 0.7})

def draw_fidelity(ax, f):
    d = f.query('dimensions<=22')
    for col, label, color in [('distance_pearson_all', 'Distance correlation', BLUE), ('trustworthiness_k10', 'Trustworthiness (10 neighbors)', ORANGE), ('neighbor_recall_k10', 'Neighbor recall (10 neighbors)', '#777777')]:
        ax.plot(d.dimensions, d[col], label=label, color=color)
    for v in [4, 10, 13]:
        ax.axvline(v, color='#DDDDDD', lw=0.8, zorder=0)
    ax.set(xlabel='Dimensions retained', ylabel='Agreement with complete 54-D MCA space', ylim=(0, 1.03), xticks=[1, 4, 10, 13, 20])
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    title(ax, 'Full-space fidelity')
    clean(ax)


class Renderer:

    def __init__(self, paths, out, dpi, font):
        self.paths = paths
        self.out = out
        self.dpi = dpi
        self.stats = {}
        self.checks = {}
        self.captions = {}
        self.records = []
        findfont(FontProperties(family=font), fallback_to_default=False)
        plt.rcParams.update({'font.family': font, 'font.size': 9, 'axes.labelsize': 10, 'axes.linewidth': 0.65, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'savefig.facecolor': 'white'})
        for name in ['panels', 'keys']:
            (out / name).mkdir(parents=True, exist_ok=True)

    def csv(self, key, index=False):
        return pd.read_csv(self.paths[key], index_col=0 if index else None, float_precision='round_trip', keep_default_na=bool(index))

    def single(self, code, draw, size=(5.8, 4.7)):
        fig, ax = plt.subplots(figsize=size, layout='constrained')
        draw(ax)
        self.save(code, fig)

    def save(self, code, fig):
        for ax in fig.axes:
            require(not re.match('^[A-H]\\s{2,}', ax.get_title(loc='left')), 'Panel letter found in title.')
        fig.savefig(self.out / 'panels' / PANEL_FILES[code], dpi=self.dpi, facecolor='white')
        plt.close(fig)
        self.records.append({'review_id': code, 'file': 'panels/' + PANEL_FILES[code], 'letters_embedded': False})

    def run(self):
        for name in ['s1', 's2', 's3', 's4', 's5', 's6']:
            print('Rendering ' + name.upper(), flush=True)
            getattr(self, name)()
        require(len(self.records) == 25, 'Expected 25 supplementary panels.')
        (self.out / 'STATISTICS.txt').write_text('Exact statistics and display checks; no new inferential tests.\n\n' + json.dumps(self.stats, indent=2, allow_nan=False) + '\n')

    def s1(self):
        s = self.csv('spectrum')
        f = self.csv('fidelity')
        p = self.csv('projections').sort_values('descriptive_rank')
        o = self.csv('omissions')
        require(len(s) == 54 and int(truth(s.in_longest_leading_run).sum()) == 13, 'MCA retention changed.')
        require(o.parallel_retained_dimensions.min() == 10, 'Omission retention changed.')

        def spectrum(ax):
            d = s.head(22)
            ax.plot(d.axis, d.eigenvalue_raw, 'o-', ms=3, color=BLUE, label='Observed')
            ax.plot(d.axis, d.null_q95, color='#777777', ls='--', label='Null 95th percentile')
            ax.axvline(13.5, color='#999999', lw=0.8, ls=':')
            ax.set(xlabel='Axis rank (Dim0 is rank 1)', ylabel='Raw eigenvalue', xticks=[1, 4, 10, 13, 20])
            ax.legend(frameon=False)
            title(ax, 'Parallel analysis')
            clean(ax)

        def inertia(ax):
            d = s.head(22)
            for col, label, color in [('raw_inertia_cumulative', 'Raw', '#8A8A8A'), ('benzecri_rate_cumulative', 'Benzécri normalized', BLUE), ('greenacre_adjusted_cumulative', 'Greenacre adjusted', ORANGE)]:
                ax.plot(d.axis, 100 * d[col], label=label, color=color)
            ax.set(xlabel='Dimensions retained', ylabel='Cumulative inertia (%)', xticks=[1, 4, 10, 13, 20])
            ax.legend(frameon=False, loc='lower right')
            title(ax, 'Cumulative inertia')
            clean(ax)

        def fidelity(ax):
            draw_fidelity(ax, f)

        def projection(ax):
            names = p.projection.str.replace('x', ' × ', regex=False)
            ax.barh(range(6), p.mean_trustworthiness, color=[BLUE if v == 'Dim0xDim3' else '#B5BDC4' for v in p.projection])
            ax.set(yticks=range(6), yticklabels=names, xlim=(0, 1), xlabel='Mean neighborhood trustworthiness')
            ax.invert_yaxis()
            title(ax, 'Six projections of the 4-D geometry')
            clean(ax)

        def distances(ax):
            x = np.arange(6)
            names = p.projection.str.replace('x', ' × ', regex=False)
            ax.plot(x, p.distance_pearson, 'o-', label='Distance Pearson r', color=BLUE)
            ax.plot(x, p.neighbor_recall_k10, 's-', label='Neighbor recall (10 neighbors)', color=ORANGE)
            ax.set(xticks=x, xticklabels=names, ylim=(0, 1.03), ylabel='Agreement with the 4-D geometry')
            ax.tick_params(axis='x', rotation=35)
            ax.legend(frameon=False, fontsize=8)
            title(ax, 'Projection fidelity')
            clean(ax)

        def retention(ax):
            d = o.set_index('held_out_participant').loc[['p01', 'p02', 'p03', 'p04', 'p05']]
            ax.bar(range(5), d.parallel_retained_dimensions, color=GRAY)
            ax.axhline(13, color=BLUE, ls='--', label='Full sample: 13-D')
            ax.axhline(10, color=ORANGE, ls=':', label='Lowest omission: 10-D')
            ax.set(xticks=range(5), xticklabels=[f'P{i + 1:02d}' for i in range(5)], ylim=(0, 17), xlabel='Participant omitted', ylabel='Dimensions retained')
            ax.legend(frameon=False, fontsize=8)
            title(ax, 'MCA refit after participant omission')
            clean(ax)
        for code, draw in zip(['S1A', 'S1C', 'S1B', 'S1D'], [spectrum, inertia, fidelity, retention]):
            self.single(code, draw)
        self.checks['MCA_retention'] = True

    def s2(self):
        e = self.csv('events')
        edges = self.csv('edges')
        ref = self.csv('flow_reference')
        require(len(e) == 256 and len(edges) == 251, 'S3 event/edge counts changed.')
        keys = ['participant_id_norm', 'hippo_id_norm']
        lookup = {tuple(v): i for i, v in enumerate(e[keys].values)}
        source = np.array([lookup[v.participant_id_norm, v.source_event] for v in edges.itertuples()])
        target = np.array([lookup[v.participant_id_norm, v.target_event] for v in edges.itertuples()])
        xy = e[['Dim0', 'Dim3']].to_numpy(float)
        delta = xy[target] - xy[source]
        lo = xy.min(0)
        hi = xy.max(0)
        pad = 0.05 * (hi - lo)
        xe = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 26)
        ye = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 26)
        x = (xe[:-1] + xe[1:]) / 2
        y = (ye[:-1] + ye[1:]) / 2

        def hist(points, weights=None):
            return np.histogram2d(points[:, 0], points[:, 1], bins=(xe, ye), weights=weights)[0].T
        background = np.log1p(gaussian_filter(hist(xy), 1.0, mode='nearest'))

        def field(mask):
            points = xy[source[mask]]
            counts = gaussian_filter(hist(points), 0.75, mode='nearest')
            v = []
            for d in range(2):
                sums = gaussian_filter(hist(points, delta[mask, d]), 0.75, mode='nearest')
                v.append(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0))
            return (counts, v)
        counts, v = field(np.ones(251, bool))
        for values, col in [(background, 'log1p_smoothed_event_count'), (counts, 'smoothed_source_count'), (v[0], 'mean_delta_Dim0'), (v[1], 'mean_delta_Dim3')]:
            close(values.ravel(), ref[col].to_numpy(float), 'S3 global field differs from main Figure 2.')
        cmap = LinearSegmentedColormap.from_list('event_density', ['#fafafa', '#dde8e8', '#a4c1c6', '#668f9c'])
        spec = [('S2A', 'context_type_primary', 'social', 'Social context', 143), ('S2B', 'context_type_primary', 'physical', 'Physical context', 64), ('S2C', 'context_type_primary', 'internal_simulated', 'Internal/simulated context', 30), ('S2D', 'context_type_primary', 'virtual_digital', 'Virtual/digital context', 14), ('S2E', 'anchor_type', 'social', 'Social anchor', 146), ('S2F', 'anchor_type', 'landmark_static', 'Static landmark anchor', 66), ('S2G', 'anchor_type', 'goal', 'Goal anchor', 18), ('S2H', 'anchor_type', 'landmark_dynamic', 'Dynamic landmark anchor', 15)]
        for code, column, value, heading, n in spec:
            mask = e.iloc[source][column].eq(value).to_numpy()
            require(mask.sum() == n, f'Unexpected support for {code}: {mask.sum()}')
            c, velocity = field(mask)
            fig = plt.figure(figsize=(5.4, 4.8))
            ratio = (x[-1] - x[0]) / (y[-1] - y[0])
            ax = fig.add_axes([0.135, 0.11, 0.8, 0.8 * 5.4 / ratio / 4.8])
            ax.contourf(x, y, background, levels=np.linspace(0, 2.1, 19), cmap=cmap, vmin=0, vmax=2.1)
            ax.contour(x, y, background, levels=np.linspace(0, 2.1, 7)[1:], colors='#527580', linewidths=0.35, alpha=0.35)
            ax.streamplot(x, y, *[np.ma.array(w, mask=c < 0.05) for w in velocity], density=1.2, color='#172d35', linewidth=0.7, arrowsize=0.85)
            ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), xlabel='Dim0', ylabel='Dim3')
            ax.set_aspect('equal')
            clean(ax)
            title(ax, heading)
            self.save(code, fig)
            self.stats[code] = {'source_transitions': int(n), 'grid': [25, 25], 'flow_sigma': 0.75, 'support_threshold': 0.05}
        fig = plt.figure(figsize=(1.65, 4.8))
        ax = fig.add_axes([0.17, 0.14, 0.16, 0.73])
        bar = ColorbarBase(ax, cmap=cmap, norm=Normalize(0, 2.1), ticks=np.arange(0, 2.11, 0.3))
        bar.set_label('log(1 + smoothed event count)')
        bar.outline.set_linewidth(0.5)
        fig.savefig(self.out / 'keys/S2_density_key.png', dpi=self.dpi, facecolor='white')
        plt.close(fig)
        self.checks['S2_matches_Figure2_global_renderer'] = True

    def s3(self):
        screen = self.csv('screen')
        eligible = screen[truth(screen.eligible)].copy()
        s = self.csv('selected').set_index('assignment').loc[AIDS]
        require(len(screen) == 2496 and len(eligible) == 2030, 'Screen counts changed.')
        for col in ['D4_explained_fraction', 'min_state_fit_silhouette']:
            eligible[col] = pd.to_numeric(eligible[col])
        diagnostics = {a: self.csv('diagnostic_' + a) for a in AIDS}
        for a, d in diagnostics.items():
            require(d.n_events.sum() == 256 and d.outgoing.sum() == 251 and (len(d) == int(a.split('_')[1][1:])), 'Basin diagnostic counts changed.')

        def tradeoff(ax):
            for dim, col in [(4, '#8BB2D0'), (2, '#D7B28B')]:
                d = eligible[eligible.dim.astype(int) == dim]
                ax.scatter(d.D4_explained_fraction, d.min_state_fit_silhouette, s=15, color=col, alpha=1, linewidth=0, label=f'{dim}-D fitting')
            offsets = [(9, -23), (10, 15), (-94, 11), (10, -26), (-90, 17)]
            for (a, r), offset in zip(s.iterrows(), offsets):
                ax.scatter(r.D4_explained_fraction, r.minimum_state_fit_silhouette, s=42, marker='o', color=COLORS[a], edgecolor=INK, lw=1, alpha=1, zorder=5)
                ax.annotate(LABELS[a], (r.D4_explained_fraction, r.minimum_state_fit_silhouette), xytext=offset, textcoords='offset points', fontsize=8, color=INK, zorder=6, bbox={'fc': 'white', 'ec': 'none', 'pad': 1.5}, arrowprops={'arrowstyle': '-', 'color': INK, 'lw': 0.45})
            ax.axhline(0, color='#858585', ls=':', lw=0.8)
            ax.set(xlabel='Fraction of 4-D variation accounted for', ylabel='Lowest basin-mean fitting-space silhouette')
            title(ax, 'Geometric tradeoffs')
            clean(ax)
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], marker='o', ms=6, ls='', mfc='white', mec=INK, label='Retained solution'))
            ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper right')

        def occupancy(ax):
            for i, a in enumerate(AIDS):
                d = diagnostics[a]
                mean = float(d.n_events.mean())
                ax.barh(i, mean, height=0.46, color='#D4E2EC' if a.startswith('D4') else '#EDDDCC', edgecolor=COLORS[a], lw=0.55, zorder=1)
                vals, ns = np.unique(d.n_events, return_counts=True)
                ax.scatter(vals, np.full(len(vals), i), s=32, color=COLORS[a], edgecolor='white', lw=0.45, zorder=3)
                for v, n in zip(vals, ns):
                    if n > 1:
                        ax.annotate(f'×{n}', (v, i), xytext=(0, 8), textcoords='offset points', ha='center', va='bottom', fontsize=8)
            ax.set(yticks=range(5), yticklabels=[LABELS[a] for a in AIDS], xlim=(0, 55), ylim=(-0.65, 4.65), xlabel='Event-states per basin-state')
            ax.invert_yaxis()
            title(ax, 'Basin-state occupancy')
            clean(ax)
            ax.legend(handles=[Patch(fc='#E0E5E8', ec='#888', lw=0.5, label='Mean'), Line2D([], [], marker='o', ls='', ms=5, color=INK, label='Basin-state')], frameon=False, loc='lower right', fontsize=8)

        def silhouette(ax):
            for i, a in enumerate(AIDS):
                d = diagnostics[a]
                ax.scatter(d.fit_silhouette, np.full(len(d), i), s=31, color=COLORS[a], edgecolor='white', lw=0.45)
            ax.axvline(0, color='#999', ls=':', lw=0.8)
            ax.set(yticks=range(5), yticklabels=[LABELS[a] for a in AIDS], xlabel='Basin-mean silhouette in the fitting space', ylim=(-0.6, 4.6))
            ax.invert_yaxis()
            title(ax, 'Basin-state separation')
            clean(ax)
        self.single('S3A', tradeoff, size=(7, 5.4))
        self.single('S3B', occupancy, size=(6.7, 4.6))
        self.single('S3C', silhouette, size=(6.7, 4.6))
        self.stats['S3'] = {'requested_configurations': 2496, 'eligible_configurations': 2030, 'mean_occupancy': {LABELS[a]: 256 / len(diagnostics[a]) for a in AIDS}, 'all_screen_circle_alpha': 1.0, 'screen_circle_area': 15, 'selected_circle_area': 42, 'all_occupancy_points_share_row_y': True}

    def s4(self):
        for code, a in zip(['S4B', 'S4A'], MAIN):
            d = self.csv('all11_pairs_' + a)
            st = self.csv('all11_stats_' + a).iloc[0]
            x = d.feature_profile_similarity_cosine.to_numpy(float)
            y = d.routing_profile_similarity_cosine.to_numpy(float)
            close(np.corrcoef(x, y)[0, 1], float(st.pearson_r), 'All11 r differs.')

            def draw(ax):
                ax.scatter(x, y, s=29, color=COLORS[a], edgecolor='white', lw=0.4)
                m, b = np.polyfit(x, y, 1)
                xx = np.array([min(x), max(x)])
                ax.plot(xx, m * xx + b, color='#4D555B', lw=1)
                ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel='Feature-profile cosine similarity', ylabel='One-step routing-profile cosine similarity')
                title(ax, LABELS[a])
                clean(ax)
            self.single(code, draw)
            self.stats[code] = {'pearson_r': float(st.pearson_r), 'nominal_two_sided_p': float(st.pearson_permutation_p_two_sided), 'basin_pairs': len(d)}

    def s5(self):
        a = AIDS[3]
        cent = self.csv('centroids_' + a, True)
        pca = self.csv('pca', True)
        align = self.csv('aligned')
        d = self.csv('event_pairs_' + a)
        v = self.csv('pca_variance')
        st = row(self.csv('event_stats_' + a), geometry_D=4, representation='full_profile')
        pr = self.csv('procrustes_stats').iloc[0]
        self.s5_centroids = cent
        require(len(cent) == 9, 'Expected nine basin-states.')
        labels = list(cent.index.astype(str))
        x = cent[['Dim0', 'Dim3']].to_numpy(float)
        y = pca.iloc[:, :2].to_numpy(float)

        def map_panel(ax, xy, names, opened):
            lim = bounds([xy])
            ax.scatter(*xy.T, s=68, facecolors='white' if opened else PALETTE[:9], edgecolors=PALETTE[:9] if opened else 'black', linewidths=1.25 if opened else 0.65, zorder=3)
            ax.set(xlim=lim[0], ylim=lim[1], xlabel=names[0], ylabel=names[1])
            ax.set_aspect('equal')
            clean(ax)
            title(ax, 'Successor projection' if opened else 'Representational projection')
            basin_labels(ax, xy, labels)
        self.single('S5A', lambda ax: map_panel(ax, x, ['Dim0', 'Dim3'], False))
        self.single('S5B', lambda ax: map_panel(ax, y, ['Successor PC1', 'Successor PC2'], True))

        def alignment(ax):
            u = align[['terrain_x', 'terrain_y']].to_numpy(float)
            v = align[['predictive_x', 'predictive_y']].to_numpy(float)
            lim = bounds([u, v], 0.22)
            for i, c in enumerate(PALETTE[:9]):
                ax.plot([u[i, 0], v[i, 0]], [u[i, 1], v[i, 1]], color='#B1B6BC', lw=0.75, zorder=1)
                ax.scatter(*u[i], s=68, c=c, edgecolor='black', lw=0.65, zorder=3)
                ax.scatter(*v[i], s=68, facecolor='white', edgecolor=c, lw=1.25, zorder=4)
            ax.set(xlim=lim[0], ylim=lim[1], xlabel='Normalized aligned coordinate 1', ylabel='Normalized aligned coordinate 2')
            ax.set_aspect('equal')
            clean(ax)
            title(ax, 'Procrustes alignment')
            basin_labels(ax, np.vstack([u, v]), labels + labels)
        self.single('S5C', alignment, size=(6.1, 5.1))
        self.distance('S5D', a, d.geometry_distance.to_numpy(float), d.predictive_distance.to_numpy(float), 'Complete successor profiles')
        close(np.corrcoef(d.geometry_distance.astype(float), d.predictive_distance.astype(float))[0, 1], float(st.pearson_r), 'S7 r differs.')
        self.stats['S5'] = {'pearson_r': float(st.pearson_r), 'nominal_p': float(st.two_sided_common_label_permutation_p), 'procrustes_disparity': float(pr.procrustes_disparity), 'procrustes_p': float(pr.lower_tail_permutation_p), 'successor_PC1_PC2_percent': float(100 * v.variance_ratio.iloc[:2].astype(float).sum())}

    def distance(self, code, a, x, y, heading, limits=None):

        def draw(ax):
            ax.scatter(x, y, s=30, color=COLORS[a], edgecolor='white', lw=0.4)
            m, b = np.polyfit(x, y, 1)
            xx = np.array([min(x), max(x)])
            ax.plot(xx, m * xx + b, color='#50585E', lw=1)
            ax.set(xlabel='4-D basin-centroid distance', ylabel='Complete successor-profile distance')
            clean(ax)
            title(ax, LABELS[a] if code.startswith('S6') else heading)
            if limits:
                ax.set(xlim=limits[0], ylim=limits[1])
        self.single(code, draw)

    def s6(self):
        pairs = []
        for a in MAIN:
            cent = self.csv('centroids_' + a, True)
            m = self.csv('visit_matrix_' + a, True)
            m = m.loc[cent.index, cent.index]
            x = pdist(cent[['Dim0', 'Dim1', 'Dim2', 'Dim3']].to_numpy(float))
            y = pdist(m.to_numpy(float))
            st = row(self.csv('visit_stats_' + a), geometry_D=4)
            close(np.corrcoef(x, y)[0, 1], float(st.pearson_r), 'Visit r differs.')
            pairs.append((a, x, y, st))
        xmax = max((x.max() for _, x, _, _ in pairs))
        ymax = max((y.max() for _, _, y, _ in pairs))
        limits = ((0, xmax * 1.08), (0, ymax * 1.08))
        for code, (a, x, y, st) in zip(['S6A', 'S6B'], pairs):
            self.distance(code, a, x, y, 'Between-visit successor', limits)
            self.stats[code] = {'r': float(st.pearson_r), 'p': float(st.permutation_p), 'visit_transitions': int(st.visit_transitions)}
        profiles = self.csv('profiles')
        profiles = profiles[profiles.geometry_D.astype(int) == 4]
        orders = self.csv('orders')
        orders = orders[orders.geometry_D.astype(int) == 4]
        names = ['P', 'I_plus_gammaP', 'M', 'M_minus_I']
        require(set(names) == set(profiles.profile), 'Unexpected profile definitions.')
        vals = np.array([[float(row(profiles, assignment_id=a, profile=p).pearson_r) for p in names] for a in AIDS])
        ps = np.array([[float(row(profiles, assignment_id=a, profile=p).permutation_p) for p in names] for a in AIDS])

        def heatmap(ax):
            cmap = LinearSegmentedColormap.from_list('purple_orange', ['#713C8A', '#F7F5F0', '#C36513'])
            im = ax.imshow(vals, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
            ax.set(xticks=range(4), xticklabels=['P', 'I + 0.90P', 'M', 'M − I'], yticks=range(5), yticklabels=[LABELS[a] for a in AIDS])
            ax.tick_params(length=0)
            title(ax, 'Successor-profile definitions')
            for (i, j), value in np.ndenumerate(vals):
                ax.text(j, i, f'{value:.3f}' + ('\n' + stars(ps[i, j]) if stars(ps[i, j]) else ''), ha='center', va='center', fontsize=10, color='white' if abs(value) > 0.7 else INK, linespacing=1.0)
            ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label='4-D distance correlation (r)', ticks=[-1, -0.5, 0, 0.5, 1])
        self.single('S6C', heatmap, size=(7, 4.7))

        def order(ax):
            for i, a in enumerate(AIDS):
                n = row(orders, assignment_id=a, order='narrated')
                r = row(orders, assignment_id=a, order='reversed')
                yy = 4 - i
                ax.hlines(yy, float(n.shuffle_null_q025), float(n.shuffle_null_q975), color='#BFC7CD', lw=5, zorder=1)
                ax.scatter(float(n.shuffle_null_mean_r), yy, s=35, marker='|', color='#667078', zorder=2)
                ax.scatter(float(n.pearson_r), yy + 0.07, s=42, marker='o', color=BLUE, edgecolor='white', lw=0.5, zorder=3)
                ax.scatter(float(r.pearson_r), yy - 0.07, s=42, marker='^', color=ORANGE, edgecolor='white', lw=0.5, zorder=3)
            ax.axvline(0, color='#AAB3BA', lw=0.7, zorder=0)
            ax.set(xlim=(-0.55, 1), ylim=(-0.5, 4.5), yticks=range(4, -1, -1), yticklabels=[LABELS[a] for a in AIDS], xlabel='4-D representational–successor distance correlation (r)')
            clean(ax)
            title(ax, 'Narrated and reversed order')
            ax.legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, label='Narrated'), Line2D([], [], marker='^', ls='', color=ORANGE, label='Reversed'), Line2D([], [], lw=5, color='#BFC7CD', label='Shuffle central 95%')], loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8)
        self.single('S6D', order, size=(7.4, 4.8))
        self.stats['S6C'] = {'profiles': names, 'r': vals.tolist(), 'nominal_label_permutation_p': ps.tolist(), 'stars': [[stars(p) for p in line] for line in ps]}
        self.stats['S6D'] = [{'solution': LABELS[a], 'narrated_r': float(row(orders, assignment_id=a, order='narrated').pearson_r), 'reversed_r': float(row(orders, assignment_id=a, order='reversed').pearson_r), 'narrated_order_shuffle_p': float(row(orders, assignment_id=a, order='narrated').order_shuffle_p_upper_tail)} for a in AIDS]
