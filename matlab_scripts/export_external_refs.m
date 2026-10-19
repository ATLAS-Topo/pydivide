function export_external_refs(which)
%EXPORT_EXTERNAL_REFS  Reference exports for the seven external Copernicus
% landscapes, so that the validation ladder can be run on nine DEMs.
%
%   export_external_refs()          % all seven
%   export_external_refs({'vermilion','blackmesa'})
%
% Six DEMs are exported whole; the Drakensberg as a 6000 x 6000 window
% (test_data/drakensberg_crop_6000.tif, the 6000 x 6000 window of the
% Copernicus tile that holds the escarpment), because the reference pipeline is
% quadratic in the number of segments and the whole 228 M-cell DEM would
% take about a day. Each landscape gets reference_data/<name>/ with the
% files export_reference.m and export_T0_nodrop.m produce; the pytest
% ladder discovers every directory that has a manifest.json.
%
% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (C) 2026 Daniel Peifer

pydivide_root = getenv('PYDIVIDE_ROOT');
if isempty(pydivide_root)
    pydivide_root = fileparts(fileparts(mfilename('fullpath')));
end
addpath(fullfile(pydivide_root, 'matlab_scripts'));
cop = fullfile(pydivide_root, 'test_data', 'copernicus');
L = {
 'vermilion',       fullfile(cop, 'Vermilion_Cliffs_Copernicus_DSM_Raw_UTM.tif');
 'blackmesa',       fullfile(cop, 'Black_Mesa_Copernicus_DSM_Raw_UTM.tif');
 'vindhya',         fullfile(cop, 'Vindhya_Range_1_Copernicus_DSM_Raw_UTM.tif');
 'araripe',         fullfile(cop, '9_Chapada_do_Araripe_Copernicus_DSM_Raw_UTM.tif');
 'siwa',            fullfile(cop, 'Siwa_El_Dabaa_Copernicus_DSM_Raw_UTM.tif');
 'serrageral',      fullfile(cop, '8_Serra_Geral_de_Goias_Copernicus_DSM_Raw_UTM.tif');
 'drakensberg6000', fullfile(pydivide_root, 'test_data', 'drakensberg_crop_6000.tif');
};
if nargin >= 1 && ~isempty(which)
    keep = ismember(L(:,1), which);
    L = L(keep, :);
end
thr_cells = 1112;   % 1e6 m^2 at 30 m, the convention of every reference export
for i = 1:size(L, 1)
    name = L{i,1}; dem = L{i,2};
    out = fullfile(pydivide_root, 'reference_data', name);
    if exist(fullfile(out, 'T0_nodrop.bin'), 'file')
        fprintf('[%s] %s: already exported\n', datestr(now, 31), name); continue
    end
    fprintf('[%s] %s: %s\n', datestr(now, 31), name, dem);
    t = tic;
    try
        export_reference(dem, out, thr_cells);
        export_T0_nodrop(dem, out, thr_cells);
        fprintf('[%s] %s done in %.0f s\n', datestr(now, 31), name, toc(t));
    catch err
        fprintf('[%s] %s FAILED after %.0f s: %s\n', datestr(now, 31), name, toc(t), err.message);
    end
end
end
