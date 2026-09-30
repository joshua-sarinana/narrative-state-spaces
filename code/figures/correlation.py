"""Deterministic analysis of the frozen 256-event dataset."""
import argparse
import hashlib
import io
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
IDS = ['D4_K10_L9_U47', 'D4_K9_L8_U49', 'D4_K6_L8_U50', 'D2_K9_L8_U50', 'D2_K8_L8_U39']
LABELS = ['4-D 10-Basin (main)', '4-D 9-Basin', '4-D 6-Basin', '2-D 9-Basin', '2-D 8-Basin']
BASIN_COUNTS = [10, 9, 6, 9, 8]
GEOMETRIES = [4, 10, 13]
GAMMA = 0.9
ALPHA = 0.05
DPI = 600
SIZE = (4.65, 3.75)
PIXELS = (2790, 2250)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_write(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f'{name}: missing columns {missing}')

def prepare_data(paths, np, pd):
    source = pd.read_csv(paths['successor_correlations'])
    overview = pd.read_csv(paths['assignment_overview'])
    components = pd.read_csv(paths['cross_solution_components'])
    holm = pd.read_csv(paths['across_geometry_holm'])
    require_columns(source, ['assignment_id', 'gamma', 'geometry_D', 'representation', 'pearson_r', 'n_pairs', 'n_permutations', 'seed'], 'correlations')
    require_columns(overview, ['assignment_id', 'events', 'edges', 'mode', 'K', 'primary_D4_gamma090_pearson_r'], 'overview')
    require_columns(components, ['assignment_id', 'geometry_D', 'pearson_r', 'gamma', 'profile', 'transition_unit', 'precision_checked_p'], 'components')
    require_columns(holm, ['assignment_id', 'basis', 'family', 'input_p', 'holm_p', 'reject_holm_at_0_05'], 'Holm results')
    selected = source[np.isclose(source.gamma, GAMMA, rtol=0, atol=1e-12) & source.representation.eq('full_profile') & source.assignment_id.isin(IDS) & source.geometry_D.isin(GEOMETRIES)].copy()
    require(len(selected) == 15, 'Expected exactly 15 complete-profile estimates at gamma 0.90.')
    require(not selected.duplicated(['assignment_id', 'geometry_D']).any(), 'Duplicate correlation rows.')
    require(np.isfinite(selected.pearson_r).all() and selected.pearson_r.between(-1, 1).all(), 'Nonfinite or invalid Pearson correlations.')
    require(selected.n_permutations.eq(10000).all() and selected.seed.eq(42).all(), 'Expected full downstream results: 10,000 permutations, seed 42.')
    overview = overview[overview.assignment_id.isin(IDS)].copy()
    require(len(overview) == 5 and (not overview.assignment_id.duplicated().any()), 'Expected five unique overview rows.')
    require(overview.events.eq(256).all() and overview.edges.eq(251).all() and overview['mode'].eq('full').all(), 'Overview must identify a full 256-event / 251-adjacency run.')
    selected_components = components[components.assignment_id.isin(IDS) & components.geometry_D.isin(GEOMETRIES)].copy()
    require(len(selected_components) == 15 and (not selected_components.duplicated(['assignment_id', 'geometry_D']).any()), 'Expected 15 unique cross-solution component rows.')
    require(np.isclose(selected_components.gamma, GAMMA, rtol=0, atol=1e-12).all() and selected_components.profile.eq('complete_M').all() and selected_components.transition_unit.eq('event_adjacency').all(), 'Cross-solution components must use complete M, gamma 0.90, event adjacency.')
    joined = selected.merge(selected_components[['assignment_id', 'geometry_D', 'pearson_r']], on=['assignment_id', 'geometry_D'], suffixes=('', '_cross'), validate='one_to_one')
    require(np.allclose(joined.pearson_r, joined.pearson_r_cross, rtol=0, atol=1e-12), 'Downstream and cross-solution correlation exports disagree; regenerate the inconsistent upstream stage.')
    selected_holm = holm[holm.basis.eq('precision_checked') & holm.family.eq('all_three_spaces') & holm.assignment_id.isin(IDS)].copy()
    require(len(selected_holm) == 5 and (not selected_holm.assignment_id.duplicated().any()), 'Expected five precision-checked, all-three-spaces Holm results.')
    require(selected_holm.holm_p.between(0, 1).all(), 'Invalid adjusted p values.')
    h = selected_holm.set_index('assignment_id')
    overview = overview.set_index('assignment_id')
    for aid, count in zip(IDS, BASIN_COUNTS):
        rows = selected[selected.assignment_id.eq(aid)]
        require(set(rows.geometry_D) == set(GEOMETRIES), f'{aid}: missing geometry.')
        require(rows.n_pairs.eq(count * (count - 1) // 2).all() and overview.loc[aid, 'K'] == count, f'{aid}: inconsistent basin or pair count.')
        r4 = rows.loc[rows.geometry_D.eq(4), 'pearson_r'].iloc[0]
        require(abs(r4 - overview.loc[aid, 'primary_D4_gamma090_pearson_r']) < 1e-12, f'{aid}: overview disagrees with the 4-D correlation.')
        maximum = selected_components.loc[selected_components.assignment_id.eq(aid), 'precision_checked_p'].max()
        require(abs(maximum - float(h.loc[aid, 'input_p'])) < 1e-12, f'{aid}: all-space input p is inconsistent with precision-checked component tests.')
        flag = str(h.loc[aid, 'reject_holm_at_0_05']).lower()
        require(flag in {'true', 'false'} and (flag == 'true') == (float(h.loc[aid, 'holm_p']) < ALPHA), f'{aid}: Holm decision is inconsistent with adjusted p.')
    selected['basin_solution'] = selected.assignment_id.map(dict(zip(IDS, LABELS)))
    selected['across_geometry_holm_p'] = selected.assignment_id.map(h.holm_p)
    selected['supported_across_geometries'] = selected.across_geometry_holm_p.lt(ALPHA)
    selected['assignment_sort'] = selected.assignment_id.map(dict(zip(IDS, range(5))))
    return selected.sort_values(['assignment_sort', 'geometry_D']).drop(columns='assignment_sort')

def make_figure(data, np, matplotlib, plt, Image):
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ['Arial', 'Nimbus Sans', 'DejaVu Sans'] if name in names))
    plt.rcParams.update({'font.family': font, 'font.size': 9, 'axes.labelsize': 9.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.65, 'xtick.major.width': 0.65, 'xtick.major.size': 3, 'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'text.color': '#202020', 'axes.labelcolor': '#202020', 'axes.edgecolor': '#404040', 'xtick.color': '#404040', 'ytick.color': '#202020'})
    colors = ['#367CA2', '#AA4499', '#CC702A']
    offsets = [0.17, 0, -0.17]
    base = np.array([4, 3, 2, 1, 0])
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0.325, 0.16, 0.625, 0.67])
    for geometry, color, offset in zip(GEOMETRIES, colors, offsets):
        values = [data.loc[data.assignment_id.eq(aid) & data.geometry_D.eq(geometry), 'pearson_r'].iloc[0] for aid in IDS]
        ax.scatter(values, base + offset, s=28, marker='o', c=color, edgecolors='white', linewidths=0.4, zorder=3)
    supported = []
    for aid, yy in zip(IDS, base):
        rows = data[data.assignment_id.eq(aid)]
        if rows.supported_across_geometries.iloc[0]:
            supported.append(aid)
            ax.scatter([rows.pearson_r.max() + 0.045], [yy], s=36, marker='$\\ast$', color='#202020', linewidths=0, zorder=4)
    ax.axvline(0, color='#8C8C8C', lw=0.8, zorder=0)
    ax.set_xlim(-0.025, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(-0.55, 4.55)
    ax.set_yticks(base, LABELS)
    ax.set_xlabel('Representational–Successor Pearson r', labelpad=7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(False)
    handles = [Line2D([], [], marker='o', mfc=color, mec='white', mew=0.4, ms=5.5, lw=0, label=f'{geometry}-D') for geometry, color in zip(GEOMETRIES, colors)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.64, 0.965), ncol=3, frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.1, title='Centroid-distance geometry', title_fontsize=8.5)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=DPI)
    plt.close(fig)
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as im:
        im.load()
        require(im.size == PIXELS, f'Unexpected PNG dimensions: {im.size}')
    return (payload, font, supported)

def legend_text():
    return 'Representational–successor correspondence across basin solutions.\n\nCircles show Pearson correlations between pairwise basin-centroid distances and distances between complete successor profiles (γ = 0.90). Each row represents one of the five basin solutions; blue, purple, and orange indicate centroid distances calculated in the 4-D, 10-D, and 13-D geometries, respectively. The dimension in each row label identifies the geometry used to construct that basin solution. Points are offset vertically for visibility.\n\n* The single asterisk beside a basin solution applies to each of its three circles, indicating support across the 4-D, 10-D, and 13-D distance geometries after Holm correction across the five solutions (adjusted p < 0.05). It denotes solution-level inference and does not indicate a difference between correlations. All estimates use the same corpus of 256 event-states.\n'
