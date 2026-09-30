"""Deterministic analysis of the frozen 256-event dataset."""
import numpy as np
FEATURES = ['primary_hipp_mech', 'secondary_hipp_mech', 'context_type_primary', 'context_type_secondary', 'anchor_type', 'frame', 'orient', 'perspective', 'landmark', 'event_meta_superordinate_theme_primary', 'event_sentiment_primary']
KEYS = ['participant_id_norm', 'hippo_id_norm']

def normalize(df):
    result = df[FEATURES].copy()
    for c in FEATURES:
        s = result[c].fillna('none').astype(str).str.strip()
        result[c] = s.mask(s.str.lower().isin({'', 'nan', 'none', 'null', 'na', 'n/a', 'nil', 'missing'}), 'none')
    return result

def fit_mca(df, reference):
    active = normalize(df)
    blocks = []
    for c in FEATURES:
        values = active[c].to_numpy()
        levels = sorted(np.unique(values))
        blocks.append((values[:, None] == np.array(levels)[None, :]).astype(float))
    z = np.column_stack(blocks)
    f = z.sum(0)
    n = len(z)
    q = len(FEATURES)
    standardized = (z - f[None, :] / n) / np.sqrt(q * f[None, :])
    u, s, _ = np.linalg.svd(standardized, full_matrices=False)
    x = np.sqrt(n) * u[:, :4] * s[None, :4]
    for j in range(4):
        if np.dot(x[:, j], reference[:, j]) < 0:
            x[:, j] *= -1
    return (x, {'n_categories': int(z.shape[1]), 'eigenvalues': (s[:6] ** 2).tolist()})
