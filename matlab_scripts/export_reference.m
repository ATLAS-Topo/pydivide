function export_reference(dem_path, out_dir, thr_cells)
%EXPORT_REFERENCE Export MATLAB DIVIDEobj reference data for pydivide validation.
%
%   export_reference(dem_path, out_dir, thr_cells)
%
%   Runs the (modified) TopoToolbox 3 DIVIDEobj pipeline on the given DEM
%   with streams defined by flowacc > thr_cells, and writes every
%   intermediate and final product needed by the pydivide validation ladder
%   as flat little-endian binary files plus a JSON
%   manifest.
%
%   The reference implementation used is the one in
%   reference_matlab\topotoolbox3-main (the modified @DIVIDEobj),
%   which is force-prepended to the path.
%
%   All index exports are 1-based MATLAB linear indices or 1-based
%   (row, col) pairs; conversion to Python conventions happens on the
%   Python side (pydivide.gridspec). All 2-D exports are transposed on
%   write so files are C-order on disk (manifest grid_order='C').
%
% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (C) 2026 Daniel Peifer

% Force the reference toolbox copy to the front of the path.
% Repository root, derived from this script's own location so the file
% works on any machine. Override with the PYDIVIDE_ROOT environment
% variable if the reference toolbox lives elsewhere.
pydivide_root = getenv('PYDIVIDE_ROOT');
if isempty(pydivide_root)
    pydivide_root = fileparts(fileparts(mfilename('fullpath')));
end
refroot = fullfile(pydivide_root, 'reference_matlab', ...
                   'topotoolbox3-main', 'toolbox');
addpath(genpath(refroot), '-begin');

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

t_all = tic;
fprintf('Reading DEM: %s\n', dem_path);
DEM = GRIDobj(dem_path);
fprintf('DEM size: %d x %d, cellsize %g\n', DEM.size(1), DEM.size(2), DEM.cellsize);

fprintf('FLOWobj (carve)...\n');
FD = FLOWobj(DEM, 'preprocess', 'carve');

fprintf('flowacc + STREAMobj (threshold %d cells)...\n', thr_cells);
A = flowacc(FD);
ST = STREAMobj(FD, A > thr_cells);

nr = FD.size(1); nc = FD.size(2);

% ---- flow edges (full network), as 1-based (r,c) pairs ------------------
[r1,c1] = ind2sub(FD.size, double(FD.ix));
[r2,c2] = ind2sub(FD.size, double(FD.ixc));
write_bin(fullfile(out_dir,'fd_edges.bin'), int64([r1 c1 r2 c2]), 'int64');
n_fd_edges = numel(r1);

% ---- stream nodes and stream edges --------------------------------------
[sr,sc] = ind2sub(FD.size, double(ST.IXgrid));
write_bin(fullfile(out_dir,'st_nodes.bin'), int64([sr sc]), 'int64');
g  = double(ST.IXgrid(ST.ix));   % giver cell linear idx
rc = double(ST.IXgrid(ST.ixc));  % receiver cell linear idx
[gr,gc]   = ind2sub(FD.size, g);
[rr,rcol] = ind2sub(FD.size, rc);
write_bin(fullfile(out_dir,'st_edges.bin'), int64([gr gc rr rcol]), 'int64');

% ---- outlets and reach labels -------------------------------------------
outlets = streampoi(ST,'outlets','ix');
[outr,outc] = ind2sub(FD.size, double(outlets));
L_nal = labelreach(ST);
STG = STREAMobj2GRIDobj(ST, L_nal);
outlet_lab = double(STG.Z(outlets));
write_bin(fullfile(out_dir,'outlets.bin'), int64([outr outc outlet_lab(:)]), 'int64');
write_bin(fullfile(out_dir,'labelreach_nal.bin'), int64(L_nal), 'int64');

fprintf('mapfromnal...\n');
Lmap = mapfromnal(FD, ST, L_nal);
write_bin(fullfile(out_dir,'L_grid.bin'), double(Lmap.Z), 'double');

% ---- DIVIDEobj, raw geometry (network=false) ----------------------------
fprintf('DIVIDEobj network=false...\n');
t0 = tic;
D0 = DIVIDEobj(FD, ST, 'network', false);
fprintf('  constructor: %.1f s\n', toc(t0));
write_bin(fullfile(out_dir,'D0_IX.bin'), double(D0.IX), 'double');

% ---- FX grid, replicated exactly from divnet ----------------------------
cs = FD.cellsize; hcs = cs/2;
[x1,y1] = ind2coord(FD, FD.ix);
[x2,y2] = ind2coord(FD, FD.ixc);
dx = x2-x1; dy = y2-y1;
mx = (x1+x2)./2; my = (y1+y2)./2;
[mix,res] = coord2ind(D0, mx, my);
FXZ = zeros(nr+1, nc+1);
FXZ(mix(res<0.5*hcs & abs(dx+dy)<cs)) = 1; % NW-SE
FXZ(mix(res<0.5*hcs & abs(dx+dy)>cs)) = 2; % NE-SW
write_bin(fullfile(out_dir,'FX_grid.bin'), int8(FXZ), 'int8');

% ---- full network: divnet + sort + divdist ------------------------------
fprintf('DIVIDEobj full network...\n');
t0 = tic;
D = DIVIDEobj(FD, ST);
fprintf('  full pipeline: %.1f s\n', toc(t0));
write_bin(fullfile(out_dir,'D_IX.bin'), double(D.IX), 'double');
write_bin(fullfile(out_dir,'D_ep.bin'), int64(D.ep), 'int64');
write_bin(fullfile(out_dir,'D_jct.bin'), int64(D.jct), 'int64');
write_bin(fullfile(out_dir,'D_jctedg.bin'), int64(D.jctedg), 'int64');
write_bin(fullfile(out_dir,'D_distance.bin'), double(D.distance), 'double');

% ---- orders (divorder_robust is the agreed reference) -------------------
for scheme = {'strahler','shreve','topo'}
    s = scheme{1};
    Ds = divorder_robust(D, s);
    write_bin(fullfile(out_dir, ['order_' s '.bin']), double(Ds.order), 'double');
end

% ---- coordinate-convention check set ------------------------------------
ixs = D0.IX(~isnan(D0.IX));
ixs = ixs(1:min(5000,numel(ixs)));
[cx,cy] = ind2coord(D0, ixs);
write_bin(fullfile(out_dir,'coordcheck.bin'), [double(ixs(:)) cx(:) cy(:)], 'double');

% ---- manifest ------------------------------------------------------------
man = struct();
man.dem_path = dem_path;
man.thr_cells = thr_cells;
man.nr = nr; man.nc = nc;
man.cellsize = FD.cellsize;
man.wf_cell = FD.wf;   % 2x3, maps 0-based (col;row;1) -> cell centers
man.wf_node = D0.wf;   % 2x3, node grid
man.grid_order = 'C';  % write_bin transposes 2-D arrays: all files are C-order
man.n_fd_edges = n_fd_edges;
man.n_st_nodes = numel(sr);
man.n_st_edges = numel(gr);
man.n_outlets = numel(outlets);
man.n_D0_IX = numel(D0.IX);
man.n_D_IX = numel(D.IX);
man.n_ep = numel(D.ep);
man.n_jct = numel(D.jct);
man.matlab_version = version;
man.exported_at = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
man.index_convention = 'MATLAB 1-based; grids column-major; node grid size (nr+1)x(nc+1)';
fid = fopen(fullfile(out_dir,'manifest.json'),'w');
fwrite(fid, jsonencode(man, 'PrettyPrint', true), 'char');
fclose(fid);

fprintf('Export complete in %.1f s -> %s\n', toc(t_all), out_dir);
end


function write_bin(p, data, cls)
%WRITE_BIN Write a matrix row-by-row (C order) as flat binary.
% MATLAB fwrite flattens column-major, so 2-D arrays are transposed first;
% every 2-D export (edge tables and grids alike) is therefore C-order on
% disk and Python reads it with reshape(n_rows, n_cols). Vectors are
% written as-is.
fid = fopen(p, 'w');
if ismatrix(data) && ~isvector(data)
    fwrite(fid, cast(data', cls), cls);
else
    fwrite(fid, cast(data(:), cls), cls);
end
fclose(fid);
end
