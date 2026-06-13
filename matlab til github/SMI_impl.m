clear all
clc
close all 

% Load data
data_lte = double(niftiread('LTE2.nii.gz'));
info_lte = niftiinfo('LTE2.nii.gz');

bvals_lte = load('LTE2.bvals'); bvals_lte = bvals_lte(:);
bvecs_lte = load('LTE2.bvecs'); bvecs_lte = bvecs_lte';

data_ste = double(niftiread('STE2.nii.gz'));
info_ste = niftiinfo('STE2.nii.gz');

bvals_ste = load('STE2.bvals'); bvals_ste = bvals_ste(:);
bvecs_ste = load('STE2.bvecs'); bvecs_ste = bvecs_ste';

% Merge data
dwi = cat(4, data_lte, data_ste);
bval = [bvals_lte; bvals_ste];

nLTE = size(data_lte,4);
nSTE = size(data_ste,4);
dirs = [bvecs_lte; zeros(nSTE,3)];
beta = [ones(nLTE,1); zeros(nSTE,1)];

% Load mask
mask = smi_create_mask(dwi, bval);
% Load noise map
sigma = smi_estimate_sigma(dwi, mask);


% Specify protocol information (no need to specify beta and TE if all
% measurements are LTE and have the same TE)
optSMI = struct();
optSMI.b = bval;
optSMI.beta = beta;
optSMI.dirs = dirs;
optSMI.TE = zeros(size(bval));
% optSMI.TE = [];
optSMI.MergeDistance = 0.1; % If []: default is 0.05 [ms/um^2], this 
% is the threshold for considering different b-values as the same shell

% Specify mask and noise map
optSMI.mask  = mask;
optSMI.sigma = sigma;


% Specify options for the fit
optSMI.compartments = {'IAS','EAS','FW'}; % The order does not matter
optSMI.D_FW = 3; % Free water diffusivity at body temperature
optSMI.NoiseBias    = 'None'; % the example data has ~ zero-mean noise
optSMI.MLTraining.bounds = [0.05, 1, 1, 0.1, 0, 50, 50, 0.05; 0.95, 3, 3, 1.2, 0.5, 150, 120, 0.99];
% The order is: [f, Da, Depar, Deperp, fw, T2a, T2e, p2] (If data has
% fixed TE then the T2a and T2e priors are simply ignored)

% Run SM fitting
tic
[out] = SMI.fit(dwi,optSMI);
t=toc;
fprintf('Time SM fit %f s\n',t)

% Load fa to make an approximate WM mask for plotting results
% [MD, FA, W, mask] = NLS_as_function('LTE2.nii.gz', 'LTE2.bvals', 'LTE2.bvecs');
% % FA_vol = zeros(nx,ny,nz);
% % FA_vol(bmask) = FA;
% nii_fa = load_untouch_nii(fullfile(pathFiles,'fa.nii'));
nii_fa = niftiread('FA_map.nii.gz');
% WM_mask = single(nii_fa > prctile(nii_fa(:), 70));
WM_mask=single(nii_fa>0.2);
% WM_mask=optSMI.mask;
WM_mask(WM_mask(:)==0)=NaN;
paramNames={'$f$','$D_\mathrm{a}\,[\mathrm{\mu m}^2/\mathrm{ms}]$','$D_\mathrm{e}^\|\,[\mathrm{\mu m}^2/\mathrm{ms}]$','$D_\mathrm{e}^\perp\,[\mathrm{\mu m}^2/\mathrm{ms}]$','$f_\mathrm{w}$','$p_2$','$p_4$'};
clims=[0 1;0 3;0 3;0 1;0 1;0 1;0 1]; slice=30; Nrows=2;


% Krot = zeros(size(out.kernel));
% 
% for p = 1:size(out.kernel,4)
%     for z = 1:size(out.kernel,3)
%         Krot(:,:,z,p) = rot90(out.kernel(:,:,z,p),-1);
%     end
% end
% 
% % K = out.kernel;
% % 
% % for p = 1:size(K,4)
% %     tmp = K(:,:,:,p);
% %     tmp(~WM_mask) = NaN;
% %     K(:,:,:,p) = tmp;
% % end
% Plot results
figure('Position',[506 225 1347 905]), SMI.plotSlices(out.kernel.*WM_mask, slice,clims,paramNames,Nrows,[],1,1)
% figure('Position',[506 225 1347 905]), SMI.plotSlices(K, slice, clims, paramNames, Nrows, [], 1, 1);
colormap hot



%%
%%% =========================
%%% SMI ROTATIONAL INVARIANTS PIPELINE (CLEAN VERSION)
%%% =========================
slice = 30;

mask2D = mask(:,:,slice);
mask_lin = mask2D(:);

kv = SMI.vectorize(out.kernel, mask)';

kv = kv(mask_lin,:); 

f      = kv(:,1);
Da     = kv(:,2);
Depar  = kv(:,3);
Deperp = kv(:,4);
fw     = kv(:,5);
p2     = kv(:,6);
p4     = kv(:,7);

x=[f Da Depar Deperp fw p2 p4];


% 3. SMI shell definition (CRITICAL)
b_shell    = out.shells(1,:);
beta_shell = out.shells(2,:);
TE_shell   = out.shells(4,:);

K0 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(0, b_shell, beta_shell, TE_shell, x, optSMI.D_FW);
K2 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(2, b_shell, beta_shell, TE_shell, x, optSMI.D_FW);
K4 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(4, b_shell, beta_shell, TE_shell, x, optSMI.D_FW);

S0_full = out.RotInvs.S0(:,:,slice,:);
S2_full = out.RotInvs.S2(:,:,slice,:);
S4_full = out.RotInvs.S4(:,:,slice,:);

S0 = reshape(S0_full, [], size(out.shells,2));
S2 = reshape(S2_full, [], size(out.shells,2));
S4 = reshape(S4_full, [], size(out.shells,2));

S0 = S0(mask_lin,:);
S2 = S2(mask_lin,:);
S4 = S4(mask_lin,:);

Sref = S0(:,1);

S0_pred = Sref .* K0;
S2_pred = Sref .* K2 .* p2;
S4_pred = Sref .* K4 .* p4;



x = S0(:);
y = S0_pred(:);

P = polyfit(x, y, 1);
x_line = linspace(min(x), max(x), 100);
yfit = polyval(P, x_line);

numBins = 200;

[counts, ~, ~, binX, binY] = histcounts2(x, y, numBins);
valid = binX > 0 & binY > 0;

binX = binX(valid);
binY = binY(valid);

x = x(valid);
y = y(valid);

density = counts(sub2ind(size(counts), binX, binY));

figure();
scatter(x, y, 10, log(density), 'filled');
set(gca,'ColorScale','log');
colormap turbo;
colorbar;
hold on;

plot(x_line, x_line, 'r-', 'LineWidth', 2);
% plot(x_line, yfit, 'k--', 'LineWidth', 1);

axis equal;
grid on;

xlabel('S_{meas}');
ylabel('S_{pred}');
title(sprintf('Correlation plot for S0 fit, r = %.4f', corr(x,y)));
xlim([-10 max(x)+10])
ylim([-10 max(y)+10])


x2 = S2(:);
y2 = S2_pred(:);

P2 = polyfit(x2, y2, 1);
x_line2 = linspace(min(x2), max(x2), 100);

numBins = 200;

[counts2, ~, ~, binX2, binY2] = histcounts2(x2, y2, numBins);

valid2 = binX2 > 0 & binY2 > 0 & ...
         binX2 <= numBins & binY2 <= numBins;

binX2 = binX2(valid2);
binY2 = binY2(valid2);

x2 = x2(valid2);
y2 = y2(valid2);

density2 = counts2(sub2ind(size(counts2), binX2, binY2));

figure();
scatter(x2, y2, 10, log(density2+1), 'filled'); % +1 avoids log(0)
set(gca,'ColorScale','log');
colormap turbo;
colorbar;
hold on;

plot(x_line2, x_line2, 'r-', 'LineWidth', 2);

axis equal;
grid on;

xlabel('S_{meas}');
ylabel('S_{pred}');
title(sprintf('S2 fit, r = %.4f', corr(x2,y2)));


x4 = S4(:);
y4 = S4_pred(:);

P4 = polyfit(x4, y4, 1);
x_line4 = linspace(min(x4), max(x4), 100);
yfit4 = polyval(P4, x_line4);

numBins = 200;

[counts4, ~, ~, binX4, binY4] = histcounts2(x4, y4, numBins);
valid4 = binX4 > 0 & binY4 > 0;

binX4 = binX4(valid4);
binY4 = binY4(valid4);

x4 = x4(valid4);
y4 = y4(valid4);

density4 = counts4(sub2ind(size(counts4), binX4, binY4));

figure();
scatter(x4, y4, 10, log(density4+1), 'filled');
set(gca,'ColorScale','log');
colormap turbo;
colorbar;
hold on;

plot(x_line4, x_line4, 'r-', 'LineWidth', 2);
% plot(x_line4, yfit4, 'k--', 'LineWidth', 1);

axis equal;
grid on;

xlabel('S_{meas}');
ylabel('S_{pred}');
title(sprintf('S4 fit, r = %.4f', corr(x4,y4)));

shell_idx = 1;

K2_single = K2(:, shell_idx);



%%% Korrelation plot %%%
% % 
% kv = SMI.vectorize(out.kernel, mask)';
% f=kv(:,1); Da=kv(:,2); Depar=kv(:,3); Deperp=kv(:,4); fw=kv(:,5); p2=kv(:,6); p4=kv(:,7);
% 
% % [b_unique, ~, b_idx] = unique(optSMI.b);
% % Nshell = numel(b_unique);   % = 3
% 
% slice = 30;
% mask_slice = mask(:,:,slice);
% 
% f_s       = f(mask_slice(:));
% Da_s      = Da(mask_slice(:));
% Depar_s   = Depar(mask_slice(:));
% Deperp_s  = Deperp(mask_slice(:));
% fw_s      = fw(mask_slice(:));
% p2_s      = p2(mask_slice(:));
% p4_s      = p4(mask_slice(:));
% 
% x_slice = [f_s Da_s Depar_s Deperp_s fw_s p2_s p4_s];
% 
% % Nshell = size(out.shells,2);
% % 
% % b_shell    = out.shells(1,:);
% % beta_shell = out.shells(2,:);
% % TE_shell   = out.shells(4,:);
% % 
% % K0 = zeros(size(x_slice,1), Nshell);
% % K2 = zeros(size(x_slice,1), Nshell);
% % K4 = zeros(size(x_slice,1), Nshell);
% % 
% % for s = 1:Nshell
% % 
% %     K0(:,s) = SMI.RotInv_Kell_wFW_b_beta_TE_numerical( ...
% %         0, ...
% %         b_shell(s), ...
% %         beta_shell(s), ...
% %         TE_shell(s), ...
% %         x_slice, ...
% %         optSMI.D_FW);
% % 
% %     K2(:,s) = SMI.RotInv_Kell_wFW_b_beta_TE_numerical( ...
% %         2, ...
% %         b_shell(s), ...
% %         beta_shell(s), ...
% %         TE_shell(s), ...
% %         x_slice, ...
% %         optSMI.D_FW);
% % 
% %     K4(:,s) = SMI.RotInv_Kell_wFW_b_beta_TE_numerical( ...
% %         4, ...
% %         b_shell(s), ...
% %         beta_shell(s), ...
% %         TE_shell(s), ...
% %         x_slice, ...
% %         optSMI.D_FW);
% % 
% % end
% 
% K0 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(0,...
%       optSMI.b, optSMI.beta, optSMI.TE, x_slice, optSMI.D_FW);
% K2 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(2,...
%       optSMI.b, optSMI.beta, optSMI.TE, x_slice, optSMI.D_FW);
% K4 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(4,...
%       optSMI.b, optSMI.beta, optSMI.TE, x_slice, optSMI.D_FW);
% 
% b_all = optSMI.b(:);   % 264×1
% Nshell = size(out.shells,2);
% b_shell = out.shells(1,:)*1000;
% beta_shell = out.shells(2,:);
% TE_shell = out.shells(4,:);
% 
% b_idx = zeros(size(b_all));
% 
% for s = 1:Nshell
%     match = (b_all == b_shell(s));
%     b_idx(match) = s;
% end
% 
% K0_shell = zeros(size(K0,1), Nshell);
% K2_shell = zeros(size(K2,1), Nshell);
% K4_shell = zeros(size(K4,1), Nshell);
% 
% for s = 1:Nshell
%     K0_shell(:,s) = mean(K0(:, b_idx == s), 2, 'omitnan');
%     K2_shell(:,s) = mean(K2(:, b_idx == s), 2, 'omitnan');
%     K4_shell(:,s) = mean(K4(:, b_idx == s), 2, 'omitnan');
% end
% 
% % K0_shell = zeros(Nvox, Nshell);
% % 
% % for s = 1:Nshell
% %     K0_shell(:,s) = mean(K0(:, b_idx == s), 2, 'omitnan');
% % end
% 
% % K0_shell = zeros(size(K0,1), Nshell);
% % K2_shell = zeros(size(K2,1), Nshell);
% % K4_shell = zeros(size(K4,1), Nshell);
% % 
% % for s = 1:5
% %     % K0_shell(:,s) = mean(K0(:, b_idx == s), 2);
% %     % K2_shell(:,s) = mean(K2(:, b_idx == s), 2);
% %     % K4_shell(:,s) = mean(K4(:, b_idx == s), 2);
% % 
% %     K0_shell(:,s) = mean(K0(:, b_idx == s), 2, 'omitnan');
% %     K2_shell(:,s) = mean(K2(:, b_idx == s), 2, 'omitnan');
% %     K4_shell(:,s) = mean(K4(:, b_idx == s), 2, 'omitnan');
% % end
% % 
% % % x = [f, Da, Depar, Deperp, fw, p2, p4];   % [Nvoxels x 7]
% % 
% % % K0 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(0, optSMI.b, optSMI.beta, optSMI.TE, x, optSMI.D_FW);
% % % K2 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(2, optSMI.b, optSMI.beta, optSMI.TE, x, optSMI.D_FW);
% % % K4 = SMI.RotInv_Kell_wFW_b_beta_TE_numerical(4, optSMI.b, optSMI.beta, optSMI.TE, x, optSMI.D_FW);
% % % 
% % % 
% % % S0_obs = out.RotInvs.S0(:,:,slice,:);
% % % S0_obs=S0_obs(mask_slice);
% % % S2_obs = out.RotInvs.S2(:,:,slice,:);
% % % S4_obs = out.RotInvs.S4(:,:,slice,:);
% % 
% % % Sref_slice  = reshape(out.RotInvs.S0(:,:,slice,:), [], 5);
% % % Sref_slice    = Sref_slice(:,1);   % or appropriate reference
% % 
% % S0_tmp = out.RotInvs.S0(:,:,slice,:);
% % S0_tmp = reshape(S0_tmp, [], 5);
% % Sref_slice = S0_tmp(mask_slice(:),1);
% % 
% S0_obs_slice = out.RotInvs.S0(:,:,slice,:);
% S0_obs_slice = reshape(S0_obs_slice, [], size(S0_obs_slice,4));
% S2_obs_slice = out.RotInvs.S2(:,:,slice,:);
% S2_obs_slice = reshape(S2_obs_slice, [], size(S2_obs_slice,4));
% S4_obs_slice = out.RotInvs.S4(:,:,slice,:);
% S4_obs_slice = reshape(S4_obs_slice, [], size(S4_obs_slice,4));
% 
% mask_lin = mask_slice(:);
% % 
% % 
% % % Sref = SMI.vectorize(out.RotInvs.S0, mask);
% % % Sref=Sref(:,1);
% % % Sref=out.RotInvs.S0(:,:,:,1);
% % % Sref=Sref(:);
% % 
% Sref_slice = out.RotInvs.S0(:,:,slice,1);
% Sref_slice = Sref_slice(mask_slice);
% S0_norm_pred = Sref_slice .* K0_shell;               % [Nvoxels x Nshells]
% S2_norm_pred = Sref_slice .* p2_s .* abs(K2_shell);   % [Nvoxels x Nshells]
% S4_norm_pred = Sref_slice .* p4_s .* abs(K4_shell);
% 
% 
% 
% x0 = S0_obs_slice(mask_lin,:);
% y0 = S0_norm_pred;
% x2 = S2_obs_slice(mask_lin,:);
% y2 = S2_norm_pred;
% x4 = S4_obs_slice(mask_lin,:);
% y4 = S4_norm_pred;
% 
% x0=x0(:);
% y0=y0(:);
% x2=x2(:);
% y2=y2(:);
% x4=x4(:);
% y4=y4(:);
% %Define the grid for density calculation
% % P0 = polyfit(x0, y0, 1);
% % 
% % x_line0 = linspace(min(x0), max(x0), 100);
% % yfit0 = polyval(P0, x_line0);
% 
% tic;
% %Define the grid for density calculation
% validPts = isfinite(x0) & isfinite(y0);
% 
% x0v = x0(validPts);
% y0v = y0(validPts);
% numBins = 500;
% 
% [counts, ~, ~, binX, binY] = histcounts2(x0v, y0v, numBins);
% 
% density = zeros(size(x0v));
% 
% ok = binX > 0 & binY > 0;
% 
% density(ok) = counts(sub2ind(size(counts), binX(ok), binY(ok)));
% 
% % numBins = 500; % Adjust number of bins based on data spread for smoother colors
% % [counts, ~, ~, binX, binY] = histcounts2(x0, y0, numBins);
% 
% %Map each point to its density
% % density = counts(sub2ind(size(counts), binX, binY));
% 
% %Plot the scatter plot with color reflecting density
% figure(); 
% hh = scatter(x0v, y0v, 10, log(density), 'filled'); % Size of points set to 15, adjust as needed
% set(gca,'ColorScale','log')
% colorbar
% xticks(0:100:max(x0v))
% yticks(0:100:max(y0v))
% colormap turbo;
% hold on;
% % plot(x_line, yfit, 'r-.', 'LineWidth', 1)
% % plot(x_line0, x_line0, 'r-', 'LineWidth', 2)
% grid on
% axis equal
% xlabel('S_{meas}'); ylabel('S_{fit}');
% title(sprintf('Correlation plot DKI NLS, \\rho = %.6f',corr(x0v,y0v)))
% xlim([-10 max(x0v)+10])
% ylim([-10 max(y0v)+10])
% toc;
% 
% 
%  for s = 1:5
% 
%     x = S0_obs_slice(mask_lin,s);
%     y = S0_norm_pred(:,s);
% 
%     valid = isfinite(x) & isfinite(y);
% 
%     figure;
%     scatter(x(valid), y(valid), 8, '.')
% 
%     hold on;
%     mn = min([x(valid); y(valid)]);
%     mx = max([x(valid); y(valid)]);
%     plot([mn mx],[mn mx],'r-')
% 
%     axis equal
%     grid on
% 
%     title(sprintf('Shell %d',s))
%     xlabel('Observed')
%     ylabel('Predicted')
% 
%     fprintf('Shell %d corr = %.6f\n', s, corr(x(valid), y(valid)));
% 
% end
%%
%%% Korrelation plot (TRUE SCATTER VERSION) %%%
% 
% S0 = out.RotInvs.S0;
% slice = round(size(S0,3)/2);
% 
% wm = WM_mask(:,:,slice) > 0;
% 
% ref = S0(:,:,slice,1);
% ref = ref(wm);
% ref = ref / median(ref(ref>0));
% 
% figure;
% 
% for s = 2:size(S0,4)
% 
%     tmp = S0(:,:,slice,s);
%     tmp = tmp(wm);
%     tmp = tmp / median(tmp(tmp>0));
% 
%     valid = isfinite(ref) & isfinite(tmp);
% 
%     subplot(2,2,s)
%     scatter(ref(valid), tmp(valid), 3, 'filled')
%     hold on
%     plot([0 max(ref)], [0 max(ref)], 'r')
%     axis equal
%     grid on
%     title(sprintf('Shell %d normalized', s))
% end

%% Helper function

function mask = smi_create_mask(dwi, bval)

% SMICREATE_MASK
% Robust brain mask for SMI pipeline (LTE + STE compatible)

% ---- Step 1: extract b0 images ----
b0_idx = bval < 50;  % adjust threshold if needed (e.g. 30–100)

if sum(b0_idx) == 0
    error('No b0 volumes found in bval.');
end

b0 = mean(dwi(:,:,:,b0_idx), 4);

% ---- Step 2: intensity thmask = imbinarize(mat2gray(b0), 0.3);threshold ----
% thr = 0.2 * max(b0(:));
% mask = b0 > thr;
mask = b0 > prctile(b0(:), 30);


% ---- Step 3: cleanup (if toolbox exists) ----
try
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 1000);
catch
    % If Image Processing Toolbox not available, skip cleanup
end

% ---- Step 4: remove tiny components (manual fallback) ----
CC = bwconncomp(mask);
numPixels = cellfun(@numel, CC.PixelIdxList);

if ~isempty(numPixels)
    [~, idx] = max(numPixels);
    cleanMask = false(size(mask));
    cleanMask(CC.PixelIdxList{idx}) = true;
    mask = cleanMask;
end

% ---- Step 5: final sanity check ----
if sum(mask(:)) < 1000
    warning('Mask may be too small — check threshold or b0 extraction.');
end

end


function sigma = smi_estimate_sigma(dwi, mask)

% SMI_ESTIMATE_SIGMA
% Robust noise standard deviation estimate for SMI pipeline
% Works with LTE + STE merged data

% ---- Step 1: estimate voxelwise noise from data variability ----
% Standard deviation across diffusion volumes
sigma_voxel = std(dwi, 0, 4);

% ---- Step 2: stabilize extreme values ----
sigma_voxel(~isfinite(sigma_voxel)) = 0;

% ---- Step 3: estimate global background noise ----
bg = ~mask;

if any(bg(:))
    sigma_global = median(sigma_voxel(bg));
else
    sigma_global = median(sigma_voxel(:));
end

% ---- Step 4: enforce stability (important for SMI) ----
sigma = sigma_voxel;

% Replace unreliable regions (outside brain or near-zero voxels)
sigma(~mask) = sigma_global;

% Avoid zeros (critical for fitting stability)
sigma(sigma == 0) = sigma_global;

% ---- Step 5: optional smoothing (if toolbox exists) ----
try
    sigma = imgaussfilt3(sigma, 1);
catch
    % skip if not available
end

end

