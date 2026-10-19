# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Maps of divide metrics, and of the basin pair a signed metric compares.

Three views, each answering a question a table cannot.

``plot_divide_values`` colours the divide network by any per-piece
column, which is the equivalent of plotting the variability of a metric
over the network and is how a user first sees where a metric is doing
something.

``plot_signed_divides`` draws the same network but adds a tick on each
piece pointing towards the side the contrast says is losing ground, so
the direction of the inference is visible on the map rather than encoded
in a column of signs.

``plot_basin_pair`` shades the two basins a signed value compares, marks
their shared divide and labels which is side A and which is side B. This
is the view that makes a sign checkable: a reader can see the two basins,
see which one is A, and decide whether the number is plausible.
"""

from __future__ import annotations

import numpy as np


def _piece_lines(D, seglength):
    """(N, 2, 2) endpoint coordinates of every exported piece."""
    seg_of_piece, p_start, p_stop = D._piece_table(seglength)
    nodes = D.network.seg_nodes
    gs = D.grid
    a = nodes[p_start]
    b = nodes[np.maximum(p_stop - 1, p_start)]
    ax, ay = _node_xy(gs, a)
    bx, by = _node_xy(gs, b)
    return np.stack([np.column_stack([ax, ay]), np.column_stack([bx, by])], axis=1)


def _node_xy(gs, node):
    w = gs.nc + 1
    i, j = np.divmod(np.asarray(node, dtype=np.int64), w)
    return gs.x0 + j * gs.cellsize, gs.y0 - i * gs.cellsize


def plot_divide_values(D, values, *, seglength: float = 300.0, ax=None,
                       cmap: str = "viridis", vmin=None, vmax=None,
                       linewidth: float = 0.8, label: str = "",
                       robust: bool = True):
    """Colour the divide network by a per-piece value.

    ``robust`` clips the colour scale to the 2nd and 98th percentiles,
    which is almost always what is wanted: divide metrics have long
    tails and a full-range scale shows one outlier and a flat map.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    v = np.asarray(values, dtype=float)
    segs = _piece_lines(D, seglength)
    if len(segs) != len(v):
        raise ValueError(f"{len(v)} values for {len(segs)} pieces at "
                         f"seglength={seglength}; pass the seglength the "
                         "values were computed at")
    ok = np.isfinite(v)
    if vmin is None or vmax is None:
        lo, hi = (np.nanpercentile(v[ok], [2, 98]) if robust and ok.any()
                  else (np.nanmin(v[ok]), np.nanmax(v[ok])) if ok.any() else (0, 1))
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8), dpi=140)
    lc = LineCollection(segs[ok], array=v[ok], cmap=cmap, linewidths=linewidth)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    cb = ax.figure.colorbar(lc, ax=ax, shrink=0.75)
    if label:
        cb.set_label(label)
    return ax


def side_a_direction(D, seglength, basin_a=None, labels=None, basin_b=None):
    """For every piece: the midpoint of the divide edge the arrow is drawn
    from and the unit vector from that midpoint into the cell on side A.

    The edge is the one nearest the middle of the piece whose two sides
    are both labelled cells of the pair the piece names: side A is
    ``basin_a`` (one label per piece, as ``across_divide_metrics``
    returns it) and side B is ``basin_b`` when given, otherwise any
    labelled cell that is not A. Without ``basin_a`` the lower of the two
    labels at the chosen edge is side A. ``labels`` defaults to
    ``D.labels``. A piece with no such edge, which happens where every
    edge of the piece borders the DEM edge, a no-data cell or a pair
    other than the one named, gets ``ok=False`` and is not drawn, so an
    arrow never points into no-data or into a basin the sign does not
    name. Returns ``(mid, unit, ok)``.
    """
    from pydivide.divmetrics import edge_cells

    L = np.asarray(D.labels if labels is None else labels).ravel()
    gs = D.grid
    _, p_start, p_stop = D._piece_table(seglength)
    n_pieces = len(p_start)
    nodes = D.network.seg_nodes
    # every edge of every piece: position k is the edge (nodes[k], nodes[k+1])
    lens = np.maximum(p_stop - p_start - 1, 0)
    piece = np.repeat(np.arange(n_pieces, dtype=np.int64), lens)
    offs = np.repeat(np.cumsum(lens) - lens, lens)
    pos = np.repeat(p_start, lens) + (np.arange(len(piece), dtype=np.int64) - offs)
    edges = np.sort(np.column_stack([nodes[pos], nodes[np.minimum(pos + 1, len(nodes) - 1)]]),
                    axis=1)
    ca, cb = edge_cells(gs, edges)
    la = np.where(ca >= 0, L[np.maximum(ca, 0)], -1)
    lb = np.where(cb >= 0, L[np.maximum(cb, 0)], -1)
    both = (la >= 0) & (lb >= 0) & (la != lb)
    if basin_a is None:
        a_is_ca = la <= lb
        good = both
    else:
        ba = np.asarray(basin_a, dtype=float)[piece]
        a_is_ca = la == ba
        a_is_cb = lb == ba
        good = both & (a_is_ca | a_is_cb)
        if basin_b is not None:
            bb = np.asarray(basin_b, dtype=float)[piece]
            good &= np.where(a_is_ca, lb == bb, la == bb)
    # the good edge nearest the middle of its piece
    mid_pos = np.repeat((p_start + p_stop - 2) / 2.0, lens)
    dist = np.where(good, np.abs(pos - mid_pos), np.inf)
    best = np.full(n_pieces, np.inf)
    np.minimum.at(best, piece, dist)
    hit = good & (dist == best[piece])
    chosen = np.full(n_pieces, -1, dtype=np.int64)
    idx = np.flatnonzero(hit)
    chosen[piece[idx][::-1]] = idx[::-1]          # the first hit of each piece wins
    ok = chosen >= 0
    sel = np.maximum(chosen, 0)
    cell = np.where(a_is_ca[sel], ca[sel], cb[sel])
    cell = np.maximum(cell, 0)
    r, c = np.divmod(cell, gs.nc)
    cx = gs.x0 + (c + 0.5) * gs.cellsize
    cy = gs.y0 - (r + 0.5) * gs.cellsize
    e = edges[sel]
    xa, ya = gs.node_xy(e[:, 0])
    xb, yb = gs.node_xy(e[:, 1])
    mid = np.column_stack([(xa + xb) / 2.0, (ya + yb) / 2.0])
    unit = np.column_stack([cx, cy]) - mid
    ln = np.hypot(unit[:, 0], unit[:, 1])
    ln[ln == 0] = 1.0
    unit = unit / ln[:, None]
    mid[~ok] = np.nan
    unit[~ok] = np.nan
    return mid, unit, ok


def plot_signed_divides(D, sgn, *, seglength: float = 300.0, ax=None,
                        tick: float = 250.0, every: int = 1,
                        convention: str = "higher_is_losing",
                        label: str = "", min_abs: float = 0.0,
                        color_a: str = "tab:blue", color_b: str = "tab:red",
                        basin_a=None, labels=None, basin_b=None):
    """Draw the direction a signed contrast implies, on the map.

    An arrow is drawn from an edge near the middle of each piece into the
    basin the contrast marks as losing ground, which is the side the
    divide is expected to migrate into. With
    ``convention='higher_is_losing'``, which is χ's convention, a
    positive sign means side A has the higher value and therefore loses.
    Hillslope relief runs the other way, so pass
    ``convention='higher_is_winning'`` for it.

    Which physical side is side A is read from the basin labels, not
    from the drawing direction of the piece: pass ``basin_a`` and
    ``basin_b`` (the columns of the table the sign came from) and the
    arrow of every piece starts on an edge that borders exactly those
    two basins and enters the one the sign names; a piece with no such
    edge (every edge on the DEM boundary or on another pair) is not
    drawn. Without them the lower label at the edge is taken as side A,
    which is the same side except where a reach label changes inside
    the piece. The tip of the arrow is ``tick`` map units from the edge;
    a long tick can cross a further divide even though the direction it
    starts in is right.

    ``min_abs`` suppresses pieces whose contrast is too weak for its sign
    to mean anything, which on a real network is most of them; setting it
    to a percentile of the population is the honest way to draw this map.
    """
    import matplotlib.pyplot as plt

    s = np.asarray(sgn, dtype=float)
    segs = _piece_lines(D, seglength)
    if len(segs) != len(s):
        raise ValueError(f"{len(s)} values for {len(segs)} pieces")
    if basin_a is not None and len(basin_a) != len(s):
        raise ValueError(f"{len(basin_a)} basin_a values for {len(s)} pieces")
    if basin_b is not None and len(basin_b) != len(s):
        raise ValueError(f"{len(basin_b)} basin_b values for {len(s)} pieces")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8), dpi=140)
    ax.plot(segs[:, :, 0].T, segs[:, :, 1].T, color="0.75", lw=0.4, zorder=1)

    mid_all, unit_all, side_ok = side_a_direction(D, seglength, basin_a, labels, basin_b)
    keep = np.isfinite(s) & (np.abs(s) > float(min_abs)) & side_ok
    idx = np.flatnonzero(keep)[::max(1, int(every))]
    if len(idx) == 0:
        return ax
    mid = mid_all[idx]
    n = unit_all[idx]                     # points into side A
    toward_a = s[idx] > 0 if convention == "higher_is_losing" else s[idx] < 0
    v = np.where(toward_a[:, None], n, -n) * float(tick)
    for i in range(len(idx)):
        ax.annotate("", xy=(mid[i, 0] + v[i, 0], mid[i, 1] + v[i, 1]),
                    xytext=(mid[i, 0], mid[i, 1]),
                    arrowprops=dict(arrowstyle="->", lw=0.6,
                                    color=color_a if toward_a[i] else color_b),
                    zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label or "direction implied by the signed contrast", fontsize=9)
    return ax


def _interior_cell(sub, lab):
    """``(row, col)`` of a cell inside the region of label ``lab``: the
    middle of the longest run of that label in the row holding most of
    its cells. ``None`` when the label is absent."""
    m = np.asarray(sub) == lab
    if not m.any():
        return None
    r = int(np.argmax(m.sum(axis=1)))
    row = m[r]
    edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    j = int(np.argmax(stops - starts))
    return r, int((starts[j] + stops[j] - 1) // 2)


def _reach_inside(L, gs, x, y, ux, uy, lab, max_reach, fallback):
    """The centre of the farthest cell along ``(ux, uy)`` from ``(x, y)``
    that still carries ``lab``, walking half a cell at a time and at most
    ``max_reach``: the arrow tip is a cell centre, strictly inside the
    basin it names, never a point on its far boundary (which the
    half-open cell lookup would still assign to the basin, on a
    one-cell-wide basin for instance). ``fallback`` is used when not even
    the first half step fits."""
    step = 0.5 * gs.cellsize
    tip = None
    d = step
    while d <= max_reach + 1e-9:
        px, py = x + ux * d, y + uy * d
        r = int(np.floor((gs.y0 - py) / gs.cellsize))
        c = int(np.floor((px - gs.x0) / gs.cellsize))
        if not (0 <= r < gs.nr and 0 <= c < gs.nc) or L[r, c] != lab:
            break
        tip = (gs.x0 + (c + 0.5) * gs.cellsize, gs.y0 - (r + 0.5) * gs.cellsize)
        d += step
    return tip if tip is not None else fallback


def plot_basin_pair(labels, pair, *, dem=None, D=None, grid=None, ax=None,
                    pad_cells: int = 40, sgn: float | None = None,
                    metric_name: str = "", seglength: float = 300.0):
    """Show the two basins a signed value compares, and which is which.

    Side A is shaded one colour, side B another, the shared divide is
    drawn between them when ``D`` is given (its edges whose two cells
    carry the two labels), and each basin is annotated with its label. If
    a signed value is given and is not zero, the title names the side the
    contrast says is losing ground under χ's convention and an arrow
    points from the divide into it; a zero is reported as the absence of
    a signed contrast and implies no direction.

    This is the view that makes a sign checkable. A column saying that
    basin 2858 stands above basin 2865 is uncheckable on its own; the
    same statement on a map is not.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    gs = grid if grid is not None else (D.grid if D is not None else None)
    if gs is None:
        raise ValueError("pass grid= or D= so the raster can be placed")
    L = np.asarray(labels)
    if L.ndim == 1:
        L = L.reshape(gs.nr, gs.nc)
    a, b = int(pair[0]), int(pair[1])
    m = (L == a) | (L == b)
    if not m.any():
        raise ValueError(f"neither basin {a} nor {b} is present in the labels")
    rows, cols = np.where(m)
    r0 = max(0, rows.min() - pad_cells)
    r1 = min(L.shape[0], rows.max() + pad_cells + 1)
    c0 = max(0, cols.min() - pad_cells)
    c1 = min(L.shape[1], cols.max() + pad_cells + 1)
    sub = L[r0:r1, c0:c1]
    ext = [gs.x0 + c0 * gs.cellsize, gs.x0 + c1 * gs.cellsize,
           gs.y0 - r1 * gs.cellsize, gs.y0 - r0 * gs.cellsize]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), dpi=140)
    if dem is not None:
        z = np.asarray(dem.z)[r0:r1, c0:c1]
        ax.imshow(z, extent=ext, origin="upper", cmap="Greys_r", alpha=0.85)
    paint = np.full(sub.shape, np.nan)
    paint[sub == a] = 0.0
    paint[sub == b] = 1.0
    ax.imshow(paint, extent=ext, origin="upper", alpha=0.45,
              cmap=ListedColormap(["tab:blue", "tab:red"]), vmin=0, vmax=1,
              interpolation="nearest")
    # each basin is annotated at a point inside it (the centre of its
    # longest run of cells in its fullest row), never at its mean cell
    # coordinate, which lies outside a concave basin
    for lab, val, col in ((a, 0.0, "tab:blue"), (b, 1.0, "tab:red")):
        rc = _interior_cell(sub, lab)
        if rc is not None:
            cx = gs.x0 + (c0 + rc[1] + 0.5) * gs.cellsize
            cy = gs.y0 - (r0 + rc[0] + 0.5) * gs.cellsize
            ax.plot(cx, cy, "o", color=col, ms=7, mec="k", zorder=5)
            ax.annotate(f"{'A' if val == 0 else 'B'}: basin {lab}", (cx, cy),
                        xytext=(8, 8), textcoords="offset points", fontsize=9,
                        color=col, weight="bold", zorder=6)
    # the shared divide: every edge of the network whose two cells carry
    # the two labels, drawn as one broken line; the anchor of the arrow is
    # one of those edges, the one nearest the middle of the set, with the
    # cell on either side of it
    anchor = None
    edges = np.asarray(getattr(D, "edge_nodes", None) if D is not None else None)
    if edges.ndim == 2 and len(edges):
        from .divmetrics import edge_cells
        ca, cb = edge_cells(gs, np.ascontiguousarray(edges, dtype=np.int64))
        flat = L.ravel()
        la = np.where(ca >= 0, flat[np.maximum(ca, 0)], -1)
        lb = np.where(cb >= 0, flat[np.maximum(cb, 0)], -1)
        on = ((la == a) & (lb == b)) | ((la == b) & (lb == a))
        if on.any():
            xa, ya = gs.node_xy(edges[on, 0])
            xb, yb = gs.node_xy(edges[on, 1])
            gap = np.full(int(on.sum()), np.nan)
            ax.plot(np.column_stack([xa, xb, gap]).ravel(),
                    np.column_stack([ya, yb, gap]).ravel(),
                    color="k", lw=1.3, zorder=4, label="shared divide")
            mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
            k = int(np.argmin((mx - mx.mean()) ** 2 + (my - my.mean()) ** 2))
            anchor = ((float(mx[k]), float(my[k])),
                      {int(la[on][k]): int(ca[on][k]), int(lb[on][k]): int(cb[on][k])})
    title = f"basin {a} (A) against basin {b} (B)"
    if sgn is not None and np.isfinite(sgn):
        if sgn == 0:
            # a zero is the absence of a signed contrast, not a direction
            title += f"\n{metric_name or 'contrast'} = 0: no signed contrast, no direction inferred"
        else:
            loser = a if sgn > 0 else b
            title += (f"\n{metric_name or 'contrast'} = {sgn:+.3f}, higher on basin {loser}"
                      f"\nthe divide is expected to move into basin {loser}")
            if anchor is not None and loser in anchor[1]:
                # from the middle of a shared edge, towards the centre of the
                # cell on its losing side, and on through that basin's cells
                # for as long as they carry the losing label, at most 15 % of
                # the view: the arrow starts on the divide and ends inside the
                # basin it names
                from .divmetrics import cell_xy
                (x_s, y_s), cells = anchor
                cxs, cys = cell_xy(gs, np.array([cells[loser]], np.int64))
                dx, dy = float(cxs[0] - x_s), float(cys[0] - y_s)
                dist = float(np.hypot(dx, dy))
                if dist > 0:
                    ux, uy = dx / dist, dy / dist
                    tip = _reach_inside(L, gs, x_s, y_s, ux, uy, loser,
                                        0.15 * max(ext[1] - ext[0], ext[3] - ext[2]),
                                        (float(cxs[0]), float(cys[0])))
                    ax.annotate("", xy=tip, xytext=(x_s, y_s),
                                arrowprops=dict(arrowstyle="->", lw=1.6, color="k"),
                                zorder=7)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return ax
