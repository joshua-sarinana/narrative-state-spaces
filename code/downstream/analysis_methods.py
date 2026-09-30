"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
FEATURES = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
FAMILIES = {'context': 'context_type_primary', 'mechanism': None, 'meta_theme': 'event_meta_superordinate_theme_primary'}
GAMMAS = [0.7, 0.8, 0.9, 0.95]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def csv(frame, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)

def square(values, labels, path):
    csv(pd.DataFrame(values, index=labels, columns=labels), path, index=True)

def bh_fdr(values):
    p = np.asarray(values, float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    vals = p[finite]
    if not len(vals):
        return out
    order = np.argsort(vals)
    q = np.minimum.accumulate((vals[order] * len(vals) / np.arange(1, len(vals) + 1))[::-1])[::-1]
    mapped = np.empty_like(vals)
    mapped[order] = np.clip(q, 0, 1)
    out[finite] = mapped
    return out

def row_normalize(counts, allow_empty=False):
    totals = counts.sum(axis=-1, keepdims=True)
    if not allow_empty:
        require((totals > 0).all(), 'A basin has no outgoing transitions; no smoothing/fabricated edge is allowed.')
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)

def count_matrix(src, dst, k):
    return np.bincount(np.asarray(src) * k + np.asarray(dst), minlength=k * k).reshape(k, k).astype(float)

def permutation_bank(k, n_perm, seed=42):
    rng = np.random.default_rng(seed)
    return np.asarray([rng.permutation(k) for _ in range(n_perm)], dtype=np.int16)

def matrix_correlation(x_matrix, y_matrix, bank, spearman=False):
    k = len(x_matrix)
    iu = np.triu_indices(k, 1)
    x, y = (x_matrix[iu], y_matrix[iu])
    require(np.std(x) > 0 and np.std(y) > 0, 'Undefined matrix correlation: constant pairwise values.')
    unique, inverse = np.unique(bank, axis=0, return_inverse=True)
    corr = (lambda a, b: float(spearmanr(a, b).statistic)) if spearman else lambda a, b: float(np.corrcoef(a, b)[0, 1])
    obs = corr(x, y)
    null_unique = np.array([corr(x, y_matrix[order][:, order][iu]) for order in unique])
    null = null_unique[inverse]
    require(np.isfinite(null).all(), 'Undefined matrix-correlation null.')
    p = (1 + np.count_nonzero(np.abs(null) >= abs(obs))) / (len(bank) + 1)
    return (obs, float(p), null)

def feature_matrix(events, k, mode):
    rows, vocabulary = ([], set())
    for row in events.itertuples(index=False):
        if mode == 'primary3':
            mechanisms = {str(row.primary_hipp_mech).strip(), str(row.secondary_hipp_mech).strip()} - {''}
            tokens = [f'mechanism_presence={v}' for v in sorted(mechanisms)]
            tokens += [f'{field}={str(getattr(row, field)).strip()}' for field in ['context_type_primary', 'event_meta_superordinate_theme_primary']]
        else:
            require(mode == 'all11', 'Unknown feature profile.')
            tokens = [f'{field}={str(getattr(row, field)).strip()}' for field in FEATURES]
        vocabulary.update(tokens)
        rows.append((int(row.state) - 1, tokens))
    vocab = sorted(vocabulary)
    lookup = {token: i for i, token in enumerate(vocab)}
    matrix = np.zeros((k, len(vocab)))
    for basin, tokens in rows:
        for token in tokens:
            matrix[basin, lookup[token]] += 1
    return (matrix, vocab)

def feature_routing(data, out, n_perm, mode):
    k, labels, events, p = (data['K'], data['labels'], data['events'], data['P'])
    counts, vocab = feature_matrix(events, k, mode)
    fs, rs = (cosine_similarity(counts), cosine_similarity(p))
    bank = permutation_bank(k, n_perm)
    r, pvalue, null_r = matrix_correlation(fs, rs, bank)
    rho, ps, null_s = matrix_correlation(fs, rs, bank, spearman=True)
    stats = {'profile_mode': mode, 'n_basins': k, 'n_pairs': k * (k - 1) // 2, 'n_feature_tokens': len(vocab), 'pearson_r': r, 'pearson_permutation_p_two_sided': pvalue, 'spearman_rho': rho, 'spearman_permutation_p_two_sided': ps, 'n_permutations': n_perm, 'seed': 42}
    csv(pd.DataFrame(counts, index=labels, columns=vocab), out / 'feature_count_matrix.csv', True)
    square(fs, labels, out / 'feature_similarity_matrix.csv')
    square(rs, labels, out / 'routing_similarity_matrix.csv')
    iu = np.triu_indices(k, 1)
    csv(pd.DataFrame({'basin_a': np.array(labels)[iu[0]], 'basin_b': np.array(labels)[iu[1]], 'feature_profile_similarity_cosine': fs[iu], 'routing_profile_similarity_cosine': rs[iu]}), out / 'pairwise_basin_similarity.csv')
    csv(pd.DataFrame({'permutation': np.arange(1, n_perm + 1), 'pearson_r_null': null_r, 'spearman_rho_null': null_s}), out / 'permutation_nulls.csv')
    csv(pd.DataFrame([stats]), out / 'feature_routing_stats.csv')

def clean_condition(value):
    s = str(value).strip()
    return None if not s or s.lower() in {'nan', 'na', 'n/a', 'null'} else s

def conditioned_payloads(transitions, family):
    if family == 'mechanism':
        values = [tuple(dict.fromkeys((v for v in map(clean_condition, pair) if v is not None))) for pair in transitions[['primary_hipp_mech', 'secondary_hipp_mech']].itertuples(index=False, name=None)]
    else:
        values = [() if (s := clean_condition(v)) is None else (s,) for v in transitions[FAMILIES[family]]]
    keep = np.array([bool(v) for v in values])
    return (transitions.loc[keep].reset_index(drop=True), [v for v in values if v])

def condition_matrices(presence, pairs, k):
    counts = np.array([np.bincount(pairs[mask], minlength=k * k).reshape(k, k) for mask in presence], float)
    return (row_normalize(counts, allow_empty=True), counts.sum(axis=2))

def condition_statistics(probs, supports, minimum=3):
    n = len(probs)
    full = cosine_similarity(probs.reshape(n, -1))
    supported = np.eye(n)
    shared = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            mask = (supports[i] >= minimum) & (supports[j] >= minimum)
            shared[i, j] = shared[j, i] = mask.sum()
            if i != j:
                value = float(cosine_similarity([probs[i][mask].ravel()], [probs[j][mask].ravel()])[0, 0]) if mask.any() else np.nan
                supported[i, j] = supported[j, i] = value
    offdiag = ~np.eye(n, dtype=bool)
    vals = supported[offdiag]
    vals = vals[np.isfinite(vals)]
    return (float(full[offdiag].mean()), float(vals.mean()) if len(vals) else np.nan, full, supported, shared)

def conditioned_routing(data, out, n_perm, family):
    transitions, values = conditioned_payloads(data['transitions'], family)
    conditions = sorted({v for vs in values for v in vs})
    conditions = [v for v in conditions if sum((v in vs for vs in values)) >= 10]
    require(len(conditions) >= 2, f'Fewer than two retained conditions in {family}.')
    presence = np.array([[v in vs for vs in values] for v in conditions])
    k, labels = (data['K'], data['labels'])
    pairs = transitions.source_state.to_numpy(int) * k + transitions.target_state.to_numpy(int)
    probs, supports = condition_matrices(presence, pairs, k)
    observed_full, observed_supported, full, supported, shared = condition_statistics(probs, supports)
    rng = np.random.default_rng(42)
    null_full, null_supported = (np.empty(n_perm), np.empty(n_perm))
    for i in range(n_perm):
        order = rng.permutation(len(transitions))
        pp, ss = condition_matrices(presence[:, order], pairs, k)
        null_full[i], null_supported[i] = condition_statistics(pp, ss)[:2]
        if (i + 1) % 2000 == 0:
            print(f"[{data['id']}] conditioned/{family}: {i + 1}/{n_perm}", flush=True)
    valid = null_supported[np.isfinite(null_supported)]
    p_full = (1 + np.count_nonzero(null_full <= observed_full)) / (n_perm + 1)
    p_supported = (1 + np.count_nonzero(valid <= observed_supported)) / (len(valid) + 1) if len(valid) and np.isfinite(observed_supported) else np.nan
    rows = []
    for test, obs, null, pval in [('full_matrix', observed_full, null_full, p_full), ('supported_source', observed_supported, valid, p_supported)]:
        rows.append({'field_key': family, 'test_family': test, 'observed_stat': obs, 'null_mean': float(null.mean()) if len(null) else np.nan, 'lower_tail_p_value': pval, 'n_valid_nulls': len(null), 'n_perm': n_perm, 'seed': 42, 'n_conditions': len(conditions), 'n_basins': k, 'min_condition_transitions': 10, 'min_source_outgoing': 3, 'status': 'ok' if np.isfinite(pval) else 'not_estimable'})
    csv(pd.DataFrame(rows), out / 'conditioned_routing_results.csv')
    csv(pd.DataFrame({'permutation': np.arange(1, n_perm + 1), 'full_matrix': null_full, 'supported_source': null_supported}), out / 'permutation_nulls.csv')
    square(full, conditions, out / 'full_matrix_cosine_similarity.csv')
    square(supported, conditions, out / 'supported_rows_cosine_similarity.csv')
    square(shared, conditions, out / 'shared_source_counts.csv')
    condition_rows = []
    for i, condition in enumerate(conditions):
        name = f'condition_{i + 1:02d}'
        square(probs[i], labels, out / f'{name}_routing_matrix.csv')
        csv(pd.DataFrame({'basin': labels, 'outgoing_transitions': supports[i].astype(int)}), out / f'{name}_row_support.csv')
        condition_rows.append({'condition_id': name, 'condition': condition, 'n_transitions': int(presence[i].sum()), 'active_source_basins': int((supports[i] > 0).sum())})
    csv(pd.DataFrame(condition_rows), out / 'condition_summary.csv')

def distances(values):
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def successor(p, gamma):
    identity = np.eye(len(p))
    a = identity - gamma * p
    m = np.linalg.solve(a, identity)
    residual = float(np.linalg.norm(a @ m - identity, ord='fro'))
    require(residual < 1e-09 and np.isfinite(m).all(), 'Invalid successor solve.')
    require((m >= -1e-12).all() and np.allclose(m.sum(1), 1 / (1 - gamma), atol=1e-10), 'Invalid successor row sums.')
    return (m, residual, float(np.linalg.cond(a)))

def center_scale(values):
    centered = values - values.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered))
    require(norm > 0, 'Zero Procrustes coordinate norm.')
    return centered / norm

def procrustes(x, y):
    x, y = (center_scale(x), center_scale(y))
    u, _, vt = np.linalg.svd(y.T @ x)
    aligned = y @ (u @ vt)
    return (float(np.sum((x - aligned) ** 2)), x, aligned)

def successor_geometry(data, out, n_perm, gamma):
    k, labels, events = (data['K'], data['labels'], data['events'])
    m, residual, cond = successor(data['P'], gamma)
    square(m, labels, out / 'successor_matrix.csv')
    pca = PCA(n_components=k - 1)
    scores = pca.fit_transform(m)
    require(np.allclose(distances(m), distances(scores), atol=1e-10, rtol=1e-10), 'Complete PCA failed distance preservation.')
    csv(pd.DataFrame(scores, index=labels, columns=[f'SuccPC{i + 1}' for i in range(k - 1)]), out / 'successor_pca_coordinates.csv', True)
    csv(pd.DataFrame({'component': np.arange(1, k), 'variance_ratio': pca.explained_variance_ratio_, 'cumulative_variance_ratio': pca.explained_variance_ratio_.cumsum()}), out / 'successor_pca_variance.csv')
    bank = permutation_bank(k, n_perm)
    np.save(out / 'common_label_permutations.npy', bank, allow_pickle=False)
    rows, all_nulls = ([], {'permutation': np.arange(1, n_perm + 1)})
    for d in [4, 10, 13]:
        dims = [f'Dim{i}' for i in range(d)]
        centroids = events.groupby('state', sort=True)[dims].mean().to_numpy(float)
        geom = distances(centroids)
        csv(pd.DataFrame(centroids, index=labels, columns=dims), out / f'centroids_D{d}.csv', True)
        square(geom, labels, out / f'geometry_distances_D{d}.csv')
        for components in [None] + list(range(1, k)):
            pred = distances(m if components is None else scores[:, :components])
            r, pval, null = matrix_correlation(geom, pred, bank)
            representation = 'full_profile' if components is None else f'PCA{components}'
            rows.append({'gamma': gamma, 'geometry_D': d, 'representation': representation, 'successor_components': k if components is None else components, 'pearson_r': r, 'two_sided_common_label_permutation_p': pval, 'n_pairs': k * (k - 1) // 2, 'n_permutations': n_perm, 'seed': 42, 'is_primary': gamma == 0.9 and d == 4 and (components is None)})
            all_nulls[f'D{d}_{representation}'] = null
            if components is None or components == 2:
                iu = np.triu_indices(k, 1)
                csv(pd.DataFrame({'basin_a': np.array(labels)[iu[0]], 'basin_b': np.array(labels)[iu[1]], 'geometry_distance': geom[iu], 'predictive_distance': pred[iu]}), out / f'pairs_D{d}_{representation}.csv')
    viz = events.groupby('state', sort=True)[['Dim0', 'Dim3']].mean().to_numpy(float)
    disparity, x, aligned = procrustes(viz, scores[:, :2])
    proc_null = np.array([procrustes(viz, scores[order, :2])[0] for order in bank])
    proc_p = (1 + np.count_nonzero(proc_null <= disparity)) / (n_perm + 1)
    all_nulls['procrustes_disparity'] = proc_null
    csv(pd.DataFrame(all_nulls), out / 'permutation_nulls.csv')
    csv(pd.DataFrame(rows), out / 'distance_correspondence.csv')
    csv(pd.DataFrame({'basin': labels, 'terrain_x': x[:, 0], 'terrain_y': x[:, 1], 'predictive_x': aligned[:, 0], 'predictive_y': aligned[:, 1]}), out / 'procrustes_aligned_coordinates.csv')
    csv(pd.DataFrame([{'gamma': gamma, 'procrustes_disparity': disparity, 'lower_tail_permutation_p': proc_p, 'inverse_residual': residual, 'condition_number': cond, 'n_permutations': n_perm}]), out / 'geometry_diagnostics.csv')

def stationary_distribution(p):
    k = len(p)
    a = np.vstack([p.T - np.eye(k), np.ones((1, k))])
    b = np.concatenate([np.zeros(k), np.ones(1)])
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    require(pi.min() >= -1e-08, 'Stationary solution has negative mass.')
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    residual = float(np.linalg.norm(pi @ p - pi))
    require(residual < 1e-08, 'Stationary residual too large.')
    multiplicity = int(np.count_nonzero(np.abs(np.linalg.eigvals(p.T) - 1) < 1e-08))
    return (pi, multiplicity, residual)

def recurrence_metrics(counts):
    p = row_normalize(counts)
    m, _, _ = successor(p, 0.9)
    pi, multiplicity, residual = stationary_distribution(p)
    return (np.array([counts.sum(0), m.mean(0), pi]), multiplicity, residual)

def recurrence(data, out, n_perm):
    k, labels = (data['K'], data['labels'])
    obs, multiplicity, residual = recurrence_metrics(data['C'])
    rng = np.random.default_rng(42)
    null = np.empty((n_perm, 3, k))
    nonunique = 0
    for i in range(n_perm):
        counts = np.zeros((k, k))
        for values in data['sequences'].values():
            shuffled = rng.permutation(values)
            counts += count_matrix(shuffled[:-1], shuffled[1:], k)
        require(counts.sum() == 251, 'Recurrence null changed transition count.')
        null[i], mult, resid = recurrence_metrics(counts)
        nonunique += mult > 1
        residual = max(residual, resid)
        if (i + 1) % 2000 == 0:
            print(f"[{data['id']}] recurrence: {i + 1}/{n_perm}", flush=True)
    result = {'basin': labels}
    null_columns = {'permutation': np.arange(1, n_perm + 1)}
    for i, name in enumerate(['incoming', 'successor_centrality', 'stationary_probability']):
        pvalues = (1 + np.count_nonzero(null[:, i] >= obs[i][None, :], axis=0)) / (n_perm + 1)
        result[name] = obs[i]
        result[name + '_null_mean'] = null[:, i].mean(0)
        result[name + '_null_sd'] = null[:, i].std(0, ddof=1)
        result[name + '_upper_tail_p'] = pvalues
        result[name + '_bh_q'] = bh_fdr(pvalues)
        for j, label in enumerate(labels):
            null_columns[f'{name}_{label}'] = null[:, i, j]
    csv(pd.DataFrame(result), out / 'basin_recurrence_significance.csv')
    csv(pd.DataFrame(null_columns), out / 'permutation_null_metrics.csv')
    (out / 'stationary_diagnostics.json').write_text(json.dumps({'observed_eigenvalue1_multiplicity': multiplicity, 'nonunique_stationary_nulls': int(nonunique), 'maximum_stationary_residual': residual, 'stationary_estimator': 'nonnegative-clipped, normalized constrained least squares; same historical convention'}, indent=2) + '\n')

def run_component(data, out, component, n_perm):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if component.startswith('feature_'):
        feature_routing(data, out, n_perm, component.removeprefix('feature_'))
    elif component.startswith('conditioned_'):
        conditioned_routing(data, out, n_perm, component.removeprefix('conditioned_'))
    elif component.startswith('geometry_'):
        successor_geometry(data, out, n_perm, int(component.removeprefix('geometry_')) / 100)
    elif component == 'recurrence':
        recurrence(data, out, n_perm)
    else:
        raise ValueError(f'Unknown component {component}')
