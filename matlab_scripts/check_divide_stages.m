function report = check_divide_stages(dem_path, thr_cells, toolbox_root, reference_root)
%CHECK_DIVIDE_STAGES  Where the DIVIDEobj pipeline loses divide edges, stage by stage.
%
%   report = check_divide_stages(dem_path, thr_cells, toolbox_root)
%   report = check_divide_stages(dem_path, thr_cells, toolbox_root, reference_root)
%
%   Builds the raw divide geometry with the DIVIDEobj constructor
%   ('network', false), then runs divnet and sort on it separately, and
%   reports the undirected divide edges after each stage, the edges each
%   stage removes or adds, and the connected components, end nodes and
%   extent of what sort removes. Everything is computed inside MATLAB with
%   TopoToolbox functions only; no pydivide code is involved, so the
%   result is independent of the reimplementation.
%
%   Routing is FLOWobj(DEM,'preprocess','carve') unless reference_root is
%   given: then the flow edges exported by export_reference.m
%   (fd_edges.bin and manifest.json in that folder) are injected, so two
%   toolbox versions can be compared on exactly the same flow network.
%
%   Measured on the 4000 x 4000 Brazil window (brazil_crop_4000.tif and
%   reference_exports_brazil4000.zip of the paper's data deposit,
%   https://doi.org/10.5281/zenodo.14748885), thr_cells = 1112, routing
%   injected:
%
%     TopoToolbox 3 1bc93e7 (2025-12-10), unpatched
%       raw 995900 | divnet 995872 | sort 978553 undirected edges
%       divnet removes 29 and adds 1; sort removes 17319 (382 segments)
%     TopoToolbox 3 main (397fd99, 2026-09-16)
%       raw 999147 | divnet 999147 | sort 982801 undirected edges
%       divnet removes 0; sort removes 16346 (359 segments)
%   In both versions sort warns that some segments could not be ordered:
%   what it removes is one connected component along the Serra do Mar,
%   segments whose ends all meet other removed segments, so no junction
%   becomes an active endpoint and the sort stops. Every edge removed at
%   main is also removed at 1bc93e7.
%
%   Inputs
%       dem_path        GeoTIFF DEM
%       thr_cells       stream threshold in cells (flowacc > thr_cells)
%       toolbox_root    the TopoToolbox 'toolbox' folder to test
%       reference_root  optional folder with manifest.json and fd_edges.bin
%
% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (C) 2026 Daniel Peifer

prior_path = path;
restore = onCleanup(@() path(prior_path));
restoredefaultpath;
addpath(genpath(toolbox_root), '-begin');
fprintf('DIVIDEobj from %s\n', which('DIVIDEobj'));

DEM = GRIDobj(dem_path);
if nargin < 4 || isempty(reference_root)
    FD = FLOWobj(DEM, 'preprocess', 'carve');
    report.routing = 'FLOWobj carve';
else
    man = jsondecode(fileread(fullfile(reference_root, 'manifest.json')));
    assert(isequal([man.nr man.nc], DEM.size), 'reference and DEM sizes differ');
    FD = FLOWobj();
    FD.size = DEM.size; FD.cellsize = DEM.cellsize; FD.wf = DEM.wf;
    FD.georef = DEM.georef; FD.type = 'single';
    fid = fopen(fullfile(reference_root, 'fd_edges.bin'), 'rb', 'ieee-le');
    e = fread(fid, [4 Inf], 'int64=>double')';
    fclose(fid);
    FD.ix = uint32(sub2ind(FD.size, e(:,1), e(:,2)));
    FD.ixc = uint32(sub2ind(FD.size, e(:,3), e(:,4)));
    clear e
    report.routing = ['injected from ' reference_root];
end
clear DEM
A = flowacc(FD);
ST = STREAMobj(FD, A > thr_cells);
clear A

D0 = DIVIDEobj(FD, ST, 'network', false);
DN = divnet(D0, FD);
DS = sort(DN);

E0 = edges_of(D0.IX); EN = edges_of(DN.IX); ES = edges_of(DS.IX);
report.raw_edges = size(E0, 1);
report.divnet_edges = size(EN, 1);
report.sort_edges = size(ES, 1);
report.divnet_removed = size(setdiff(E0, EN, 'rows'), 1);
report.divnet_added = size(setdiff(EN, E0, 'rows'), 1);
lost = setdiff(EN, ES, 'rows');
report.sort_removed = size(lost, 1);
report.sort_added = size(setdiff(ES, EN, 'rows'), 1);
report.divnet_segments = nnz(isnan(DN.IX));
report.sort_segments = nnz(isnan(DS.IX));
report.sort_removed_components = 0;
if ~isempty(lost)
    [nodes, ~, k] = unique(lost(:));
    k = reshape(k, [], 2);
    G = graph(k(:,1), k(:,2));
    report.sort_removed_components = max(conncomp(G));
    report.sort_removed_nodes = numel(nodes);
    report.sort_removed_degree3 = nnz(degree(G) == 3);
    [r, c] = ind2sub(DN.size, nodes);
    report.sort_removed_rows = [min(r) max(r)];
    report.sort_removed_cols = [min(c) max(c)];
end
fprintf(['undirected edges: raw %d | divnet %d | sort %d\n' ...
         'divnet removes %d, adds %d | sort removes %d, adds %d\n' ...
         'segments after divnet %d, after sort %d; sort-removed components %d\n'], ...
    report.raw_edges, report.divnet_edges, report.sort_edges, report.divnet_removed, ...
    report.divnet_added, report.sort_removed, report.sort_added, report.divnet_segments, ...
    report.sort_segments, report.sort_removed_components);
end

function E = edges_of(IX)
% Undirected, de-duplicated node pairs of a NaN-separated node list.
a = IX(1:end-1); b = IX(2:end);
ok = ~isnan(a) & ~isnan(b);
E = unique(sort([a(ok) b(ok)], 2), 'rows');
end
