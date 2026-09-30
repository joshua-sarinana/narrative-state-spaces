"""Deterministic analysis of the frozen 256-event dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from numerics import require
FEATURES = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
FAMILIES = {'context': 'context_type_primary', 'mechanism': None, 'meta_theme': 'event_meta_superordinate_theme_primary'}

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
    if spearman:
        x = rankdata(x)
        ranked = np.zeros_like(y_matrix)
        ranked[iu] = rankdata(y)
        ranked[iu[1], iu[0]] = ranked[iu]
        y_matrix = ranked
        y = y_matrix[iu]
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
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
                if mask.any():
                    a, b = (probs[i][mask].ravel(), probs[j][mask].ravel())
                    an, bn = (np.linalg.norm(a), np.linalg.norm(b))
                    value = float(np.dot(a / an, b / bn)) if an > 0 and bn > 0 else 0.0
                else:
                    value = np.nan
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
