"""Two-sided Mann-Whitney on every row of a matrix, equal to scipy.stats.mannwhitneyu(x, y) row by row: exact when the
row has no ties and an arm has <= 8 values, asymptotic otherwise. The permutation FDR needs millions of these tests.
"""
import numpy as np
from scipy import stats
from scipy.stats._mannwhitneyu import _MWU

exact_sf = {}


def average_ranks(xy):
    """Average ranks per row, the tie term sum(t**3 - t) and whether the row has ties."""
    n = xy.shape[1]
    order = np.argsort(xy, axis=1, kind='stable')
    xs = np.take_along_axis(xy, order, axis=1)
    new = np.empty(xs.shape, dtype=bool)
    new[:, 0] = True
    np.not_equal(xs[:, 1:], xs[:, :-1], out=new[:, 1:])
    position = np.arange(n, dtype=np.int64)[None, :]
    start = np.maximum.accumulate(np.where(new, position, -1), axis=1)
    last = np.empty_like(new)
    last[:, :-1] = new[:, 1:]
    last[:, -1] = True
    end = np.minimum.accumulate(np.where(last, position, n)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty(xy.shape, dtype=np.float64)
    np.put_along_axis(ranks, order, (start + end) / 2.0 + 1.0, axis=1)
    size = (end - start + 1).astype(np.int64)
    return ranks, np.where(new, size ** 3 - size, 0).sum(axis=1), (np.where(new, size, 1) > 1).any(axis=1)


def mw_two_sided(F, is_mt):
    """p value per row of F (rows x patients); is_mt selects the first arm."""
    n1, n2 = int(is_mt.sum()), int((~is_mt).sum())
    n = n1 + n2
    xy = np.ascontiguousarray(F[:, np.r_[np.where(is_mt)[0], np.where(~is_mt)[0]]])
    ranks, tie_term, has_ties = average_ranks(xy)
    u1 = ranks[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2.0
    u = np.maximum(u1, n1 * n2 - u1)
    s = np.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    with np.errstate(divide='ignore', invalid='ignore'):
        p = stats.norm.sf((u - n1 * n2 / 2.0 - 0.5) / s)
    if not (n1 > 8 and n2 > 8):
        if (n1, n2) not in exact_sf:
            exact_sf[n1, n2] = np.asarray(_MWU(n1, n2).sf(np.arange(n1 * n2 + 1)), dtype=np.float64)
        p[~has_ties] = exact_sf[n1, n2][np.rint(u[~has_ties]).astype(np.int64)]
    return np.clip(p * 2.0, 0.0, 1.0)
