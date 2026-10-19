function check_divnet_edge_loss(dem_path, thr_cells, toolbox_root)
%CHECK_DIVNET_EDGE_LOSS  Edges the DIVIDEobj constructor produces but divnet drops.
%
%   check_divnet_edge_loss(dem_path, thr_cells, toolbox_root)
%
%   Runs the DIVIDEobj constructor twice on one DEM -- once with
%   'network', false (the raw divide geometry, D0) and once in full (the
%   final network, D) -- and reports how many undirected divide edges are
%   present in D0.IX but absent from D.IX, how many connected components
%   they form, and where they are. Everything is computed inside MATLAB
%   with TopoToolbox functions only; no pydivide code is involved, so
%   the result is independent of the reimplementation.
%
%   Measured with this script on 2026-09-01, TopoToolbox 3 at commit
%   fc12241 (main, 2026-07-28), on a 4000 x 4000 window of the 30 m
%   FathomDEM mosaic of south-eastern Brazil (brazil_crop_4000.tif in the
%   paper's data deposit), thr_cells = 1112:
%
%       D0 edges 999147 | D edges 982801 | in D0 not in D: 16346
%       lost-edge components: 1, largest has 16338 nodes
%       endorheic flags: 0 of 1065639 segments
%
%   At commit 1bc93e7 (2025-12-10) the same statistic is 29 edges.
%
%   Inputs
%       dem_path      GeoTIFF DEM
%       thr_cells     stream threshold in cells (flowacc > thr_cells)
%       toolbox_root  path to the TopoToolbox 'toolbox' directory to test
%                     (e.g. a fresh clone of TopoToolbox/topotoolbox3)
%
% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (C) 2026 Daniel Peifer

addpath(genpath(toolbox_root), '-begin');

DEM = GRIDobj(dem_path);
FD  = FLOWobj(DEM, 'preprocess', 'carve');
A   = flowacc(FD);
ST  = STREAMobj(FD, A > thr_cells);

D0 = DIVIDEobj(FD, ST, 'network', false);
D  = DIVIDEobj(FD, ST);

E0 = edges_of(D0.IX);
E1 = edges_of(D.IX);
lost = setdiff(E0, E1, 'rows');
fprintf('D0 edges %d | D edges %d | in D0 not in D: %d | in D not in D0: %d\n', ...
    size(E0,1), size(E1,1), size(lost,1), size(setdiff(E1,E0,'rows'),1));

if ~isempty(lost)
    G = graph(lost(:,1), lost(:,2));
    comp = conncomp(G);
    nz = comp(unique(lost(:)));
    counts = histcounts(nz, 0.5:1:max(nz)+0.5);
    fprintf('lost-edge components: %d, largest has %d nodes\n', numel(unique(nz)), max(counts));
    [r, c] = ind2sub(D.size, lost(:,1));
    fprintf('lost-edge node bbox rows %d-%d cols %d-%d (1-based, node grid)\n', min(r), max(r), min(c), max(c));
end
if isprop(D, 'endo') && ~isempty(D.endo)
    fprintf('endorheic flags: %d of %d segments; D.ep %d, D.jct %d\n', nnz(D.endo), numel(D.endo), numel(D.ep), numel(D.jct));
end
end

function E = edges_of(IX)
% Undirected, de-duplicated node pairs of a NaN-separated node list.
a = IX(1:end-1); b = IX(2:end);
ok = ~isnan(a) & ~isnan(b);
E = unique(sort([a(ok) b(ok)], 2), 'rows');
end
