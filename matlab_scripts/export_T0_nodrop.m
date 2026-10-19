function export_T0_nodrop(dem_path, out_dir, thr_cells)
%EXPORT_T0_NODROP Re-run the DIVIDEobj constructor geometry pipeline and
% export the deduplicated edge list BEFORE the "avoid single nodes" drop.
%
% Purpose: machine-classification of edge-set discrepancies.
% An edge present here but absent from D0.IX was removed by the
% order-dependent single-edge-run drop; an edge absent here as well would
% point to the GRIDobj2polygon defect instead.
%
% The geometry section below (from "DEM = GRIDobj" to the removal of
% redundant NaN separators) reproduces, line for line, the constructor
% of the reference implementation so that its edge list can be exported
% before the single-edge-run drop:
%
%   TopoToolbox 3, @DIVIDEobj/DIVIDEobj.m, commit 1bc93e7 (2025-12-10),
%   https://github.com/TopoToolbox/topotoolbox3
%   Copyright (C) Dirk Scherler and Wolfgang Schwanghart,
%   GNU General Public License v3 or later.
%
% The adaptation is the early stop and the binary export; the four local
% patches to the reference (Supplement S1 of the paper) do not touch these
% lines. See THIRD_PARTY.md at the repository root.
%
% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (C) 2026 Daniel Peifer (export and adaptation)

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

DEM = GRIDobj(dem_path);
FD = FLOWobj(DEM, 'preprocess', 'carve');
A = flowacc(FD);
ST = STREAMobj(FD, A > thr_cells);

hcs = FD.cellsize/2;

% D0 only provides the node-grid coord2ind/ind2coord frame
D0 = DIVIDEobj(FD, ST, 'network', false);

% ---- constructor replication (verbatim semantics) -----------------------
outlets = streampoi(ST,'outlets','ix');
L = labelreach(ST);
STG = STREAMobj2GRIDobj(ST,L);
L = mapfromnal(FD,ST,L);
outlet_gridval = STG.Z(outlets);

MS = GRIDobj2polygon(L);
[MS.outlet_id] = deal(nan);
[MS.endo] = deal(false);
for i = 1 : numel(MS)
    if not(isnan(MS(i).X(end)))
        MS(i).X = [MS(i).X;NaN];
        MS(i).Y = [MS(i).Y;NaN];
    end
    outix = ismember(outlet_gridval,MS(i).gridval);
    if sum(outix)>0
        MS(i).outlet_id = outlets(outix);
    end
end

st_giver = ST.IXgrid(ST.ix);
st_receiver = ST.IXgrid(ST.ixc);
[x1,y1] = ind2coord(FD,st_giver);
[x2,y2] = ind2coord(FD,st_receiver);
mx = (x1+x2)./2;
my = (y1+y2)./2;

for i = 1 : numel(MS)
    tx = MS(i).X;
    ty = MS(i).Y;

    tmx = (tx(1:end-1)+tx(2:end))./2;
    tmy = (ty(1:end-1)+ty(2:end))./2;
    ix = ismember([tmx,tmy],[mx,my],'rows');
    iy = find(ix);
    [~,isort] = sort([(1:numel(tx))';iy]);
    tx = [tx;nan(numel(iy),1)];
    ty = [ty;nan(numel(iy),1)];
    tx = tx(isort);
    ty = ty(isort);

    ix = ismember([tx,ty],[mx,my],'rows');
    iy = find(ix);
    [~,isort] = sort([(1:numel(tx))';repmat(iy,2,1)]);
    tx = [tx;nan(numel(iy),1);tx(iy)];
    ty = [ty;nan(numel(iy),1);ty(iy)];
    tx = tx(isort);
    ty = ty(isort);

    isendo = false;
    if not(isnan(MS(i).outlet_id))
        [outx,outy] = ind2coord(STG,MS(i).outlet_id);
        fx = [outx-hcs,outx-hcs,outx+hcs,outx+hcs];
        fy = [outy-hcs,outy+hcs,outy-hcs,outy+hcs];
        fi = nan(size(fx));
        fi(:) = coord2ind(D0,fx,fy);
        tix = coord2ind(D0,tx,ty);
        ix = ismember(fi,tix);
        isendo = not(any(ix,2));
        if (isendo)
            warning('found endorheic drainage')
        else
            iy = ismember([tx,ty],[fx(ix)' fy(ix)'],'rows');
            tx(iy) = nan;
            ty(iy) = nan;
        end
    end

    IX = isnan(tx);
    JX = [IX(2:end);true];
    IJ = logical(min([IX JX],[],2));
    tx = tx(not(IJ));
    ty = ty(not(IJ));

    if not(isendo)
        ix = find(isnan(tx),1,'first');
        if not(isempty(ix))
            tx = [tx(ix+1:end);tx(1:ix)];
            ty = [ty(ix+1:end);ty(1:ix)];
        end
    end

    dtx = tx(1:end-1)-tx(2:end);
    dty = ty(1:end-1)-ty(2:end);
    ix = not(dtx==0 & dty==0);
    tx = [tx(ix);NaN];
    ty = [ty(ix);NaN];

    MS(i).X = tx;
    MS(i).Y = ty;
end

I = coord2ind(D0,vertcat(MS.X),vertcat(MS.Y));

T = [I(1:end-1),I(2:end)];
T(sum(isnan(T),2)==1,:) = NaN;
sT = sort(T,2);
[~,IX,~] = unique(sT,'rows','stable');
T0 = nan(size(T));
T0(IX,:) = T(IX,:);

IX = (T0(:,1)-T0(:,2))==0;
T0(IX,:) = NaN;

IX = isnan(T0(:,1));
JX = [IX(2:end);true];
IJ = logical(min([IX JX],[],2));
T0 = [T0(not(IJ),:);NaN NaN];

% ---- export T0 (pre-drop edge list) -------------------------------------
if ~exist(out_dir,'dir'); mkdir(out_dir); end
fid = fopen(fullfile(out_dir,'T0_nodrop.bin'),'w');
fwrite(fid, double(T0'), 'double');
fclose(fid);
fprintf('T0 rows: %d (of which %d edge rows)\n', size(T0,1), sum(~isnan(T0(:,1))));
end
