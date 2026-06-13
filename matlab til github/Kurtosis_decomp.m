%%%%% NLS Kurtosis (anisotropic and isotropic) %%%%%
clear all
clc
close all 

data_lte = double(niftiread('LTE2.nii.gz'));
info_lte = niftiinfo('LTE2.nii.gz');

bvals_lte = load('LTE2.bvals'); bvals_lte = bvals_lte(:);
bvecs_lte = load('LTE2.bvecs'); bvecs_lte = bvecs_lte';

data_ste = double(niftiread('STE2.nii.gz'));
info_ste = niftiinfo('STE2.nii.gz');

bvals_ste = load('STE2.bvals'); bvals_ste = bvals_ste(:);
bvecs_ste = load('STE2.bvecs'); bvecs_ste = bvecs_ste';

assert(length(bvals_lte) == size(data_lte,4), 'LTE mismatch');
assert(length(bvals_ste) == size(data_ste,4), 'STE mismatch');

N = length(bvals_lte);

b_matrix = zeros(N, 6);
b_tensor = zeros(N, 15);

for i = 1:N
    bx = bvecs_lte(i,1);
    by = bvecs_lte(i,2);
    bz = bvecs_lte(i,3);
    b  = bvals_lte(i);

b_matrix(i,:) = [
    b*bx^2,
    b*by^2,
    b*bz^2,
    2*b*bx*by,
    2*b*bx*bz,
    2*b*by*bz];


b_tensor(i,:) = [
    b^2*bx^4,
    b^2*by^4,
    b^2*bz^4,
    4*b^2*bx^3*by,
    4*b^2*by^3*bx,
    4*b^2*bx^3*bz,
    4*b^2*bz^3*bx,
    4*b^2*by^3*bz,
    4*b^2*bz^3*by,
    6*b^2*bx^2*by^2,
    6*b^2*bx^2*bz^2,
    6*b^2*by^2*bz^2,
    12*b^2*bx^2*by*bz,
    12*b^2*by^2*bx*bz,
    12*b^2*bz^2*bx*by];
end

[nx, ny, nz, ~] = size(data_lte);

%%%%% Model %%%%%
% options = optimoptions('lsqcurvefit', ...
%     'Display','off', ...
%     'MaxIterations',20);
% 
% iso_model = @(p,b) p(1).*exp(-b.*p(2) + (1/6).*b.^2.*(p(2).^2).*p(3));

function [F,J] = iso_model_jac(p,b)

    % Parameters
    S0 = p(1);
    D  = p(2);
    K  = p(3);

    % Exponential term
    E = exp(-b.*D + (1/6).*b.^2.*D.^2.*K);

    % Signal model
    F = S0 .* E;

    % Analytical Jacobian
    if nargout > 1

        J = zeros(length(b),3);

        % dS/dS0
        J(:,1) = E;

        % dS/dD
        J(:,2) = F .* (-b + (1/3).*b.^2.*D.*K);

        % dS/dK
        J(:,3) = F .* ((1/6).*b.^2.*D.^2);

    end
end

options = optimoptions('lsqcurvefit', ...
    'Display','off', ...
    'SpecifyObjectiveGradient',true, ...
    'MaxIterations',20, ...
    'FunctionTolerance',1e-6, ...
    'StepTolerance',1e-6);

%%%%% Mask %%%%%
b0 = data_lte(:,:,:,1);
th = prctile(b0(:), 90);
bmask = b0 > 0.3 * th;


% reshape
data_lte_1D = permute(data_lte,[4,1,2,3]);
data_ste_1D = permute(data_ste,[4,1,2,3]);

% apply mask ONCE
data_lte_1D = data_lte_1D(:, bmask);
data_ste_1D = data_ste_1D(:, bmask);

% now safe
nVox = size(data_lte_1D,2);


%%%%% Maps %%%%%
K_tot_map = zeros(nVox,1);
K_iso_map = zeros(nVox,1);
K_aniso_map = zeros(nVox,1);

% S_meas_lte_all = zeros(length(uniqueB_lte), nVox);
%S_meas_lte = zeros(length(bvals_lte), nVox);
S_fit_lte  = zeros(length(bvals_lte), nVox);
% S_fit_lte_all  = zeros(length(uniqueB_lte), nVox);

% S_meas_ste_all = zeros(length(uniqueB_ste), nVox);
%S_meas_ste = zeros(length(bvals_ste), nVox);
S_fit_ste  = zeros(length(bvals_ste), nVox);
% S_fit_ste_all  = zeros(length(uniqueB_ste), nVox);

%%%%% Grouping %%%%%
uniqueB_lte = unique(bvals_lte);
shell_idx_lte = arrayfun(@(x) find(bvals_lte==x), uniqueB_lte, 'UniformOutput', false);

uniqueB_ste = unique(bvals_ste);
shell_idx_ste = arrayfun(@(x) find(bvals_ste==x), uniqueB_ste, 'UniformOutput', false);

%%%%% Fit %%%%%
parfor idx = 1:nVox

    S_lte = data_lte_1D(:, idx);
    S_ste = data_ste_1D(:, idx);

    S_lte(S_lte <= 0) = eps;
    S_ste(S_ste <= 0) = eps;

    S_avg_lte = zeros(length(uniqueB_lte), 1);
    S_avg_ste = zeros(length(uniqueB_ste), 1);

    for j = 1:length(uniqueB_lte)
        S_avg_lte(j) = mean(S_lte(shell_idx_lte{j}));
    end

    for j = 1:length(uniqueB_ste)
        S_avg_ste(j) = mean(S_ste(shell_idx_ste{j}));
    end

    % S_avg_lte = S_avg_lte / S_avg_lte(1);
    % S_avg_ste = S_avg_ste / S_avg_ste(1);

    % p_lte = lsqcurvefit(iso_model, ...
    %     [S_avg_lte(1), 1e-3, 1], ...
    %     uniqueB_lte, S_avg_lte, ...
    %     [0 0 -2], [Inf 0.003 5], options);
    % 
    % p_ste = lsqcurvefit(iso_model, ...
    %     [S_avg_ste(1), 1e-3, 1], ...
    %     uniqueB_ste, S_avg_ste, ...
    %     [0 0 -2], [Inf 0.003 5], options);

    p_lte = lsqcurvefit(@(p,b) iso_model_jac(p,b), ...
    [S_avg_lte(1), 1e-3, 1], ...
    uniqueB_lte, ...
    S_avg_lte, ...
    [0 0 -2], ...
    [Inf 0.003 5], ...
    options);

    p_ste = lsqcurvefit(@(p,b) iso_model_jac(p,b), ...
    [S_avg_ste(1), 1e-3, 1], ...
    uniqueB_ste, ...
    S_avg_ste, ...
    [0 0 -2], ...
    [Inf 0.003 5], ...
    options);
    
    if any(~isfinite(p_lte)) || any(~isfinite(p_ste))
        continue
    end
    if p_lte(2) <= 0 || p_ste(2) <= 0
        continue
    end

    % Korrelation plot
    % S_meas_lte(:,idx) = S_lte;
    % S_meas_ste(:,idx) = S_ste;
    % S_fit_lte(:, idx) = iso_model_jac(p_lte, bvals_lte);
    % S_fit_ste(:, idx) = iso_model_jac(p_ste, bvals_ste);
    [F_lte, ~] = iso_model_jac(p_lte, bvals_lte);
    [F_ste, ~] = iso_model_jac(p_ste, bvals_ste);
    
    S_fit_lte(:, idx) = F_lte;
    S_fit_ste(:, idx) = F_ste;

    K_tot_map(idx) = p_lte(3);
    K_iso_map(idx) = p_ste(3);
    K_aniso_map(idx) = p_lte(3) - p_ste(3);

end

%%%%% Reconstruct volume %%%%%
K_tot_vol = zeros(nx,ny,nz);
K_iso_vol = zeros(nx,ny,nz);
K_aniso_vol = zeros(nx,ny,nz);

K_tot_vol(bmask) = K_tot_map;
K_iso_vol(bmask) = K_iso_map;
K_aniso_vol(bmask) = K_aniso_map;

slice = round(nz/2);

K_tot_plot = K_tot_vol(:,:,slice);
K_iso_plot = K_iso_vol(:,:,slice);
K_aniso_plot = K_aniso_vol(:,:,slice);

K_tot_plot(~bmask(:,:,slice)) = NaN;
K_iso_plot(~bmask(:,:,slice)) = NaN;
K_aniso_plot(~bmask(:,:,slice)) = NaN;

%%%%% Plot %%%%%
figure;
imagesc(rot90(K_tot_plot,-1),[0 2]);
axis image off; colormap turbo; colorbar;
title(['K_{tot} - Slice ', num2str(slice)]);


figure;
imagesc(rot90(K_iso_plot,-1),[0 2]);
axis image off; colormap turbo; colorbar;
title(['K_{iso} - Slice ', num2str(slice)]);

figure;
imagesc(rot90(K_aniso_plot, -1),[0 2]);
axis image off; colormap turbo; colorbar;
title(['K_{aniso} - Slice ', num2str(slice)]);

%%
b0 = mean(data_lte(:,:,:,bvals_lte==0),4);
mask = b0 > 0.2*max(b0(:));

nii_fa = niftiread('FA_map.nii.gz');
WM_mask=single(nii_fa>0.3);
WM_mask(WM_mask(:)==0)=NaN;

K_tot_vals = K_tot_vol(mask);
K_iso_vals = K_iso_vol(mask);
K_aniso_vals = K_aniso_vol(mask);


%%
%%%% Korrelation plot %%%%%


%%% LTE %%%%
%%% --- Extract single slice ---
% S_measss_lte = data_lte_1D';   % now: voxels × gradients
% % S_fit_lte  = S_fit_lte;      % voxels × gradients
% x_lte = S_measss_lte(:);
% y_lte = S_fit_lte(:);
[idx_x, idx_y, idx_z] = ind2sub([nx, ny, nz], find(bmask));
S_fit_lte_vol = zeros(nx, ny, nz, length(bvals_lte));
S_fit_ste_vol = zeros(nx, ny, nz, length(bvals_ste));
for i = 1:nVox
    S_fit_lte_vol(idx_x(i), idx_y(i), idx_z(i), :) = S_fit_lte(:,i);
    S_fit_ste_vol(idx_x(i), idx_y(i), idx_z(i), :) = S_fit_ste(:,i);
end


slice = round(size(data_lte,3)/2);
%data_slice=data_1D';
% data_slice_lte = squeeze(data_lte(:,:,slice,:));   % (x,y,g)
% S_fit_slice_lte  = squeeze(S_fit_lte(:,:,slice,:));  % (x,y,g)
% x_lte=data_slice_lte(:);
% y_lte=S_fit_slice_lte(:);

% --- LTE slice ---
data_slice_lte = squeeze(data_lte(:,:,slice,:));
fit_slice_lte  = squeeze(S_fit_lte_vol(:,:,slice,:));

x_lte = data_slice_lte(:);
y_lte = fit_slice_lte(:);

valid_lte = isfinite(x_lte) & isfinite(y_lte) & x_lte > 0 & y_lte > 0;
x_lte = x_lte(valid_lte);
y_lte = y_lte(valid_lte);


P_lte = polyfit(x_lte, y_lte, 1);

x_line_lte = linspace(min(x_lte), max(x_lte), 100);
yfit_lte = polyval(P_lte, x_line_lte);

% Define the grid for density calculation

tic;
% Define the grid for density calculation

numBins = 500; % Adjust number of bins based on data spread for smoother colors
[counts, ~, ~, binX, binY] = histcounts2(x_lte, y_lte, numBins);

% Map each point to its density
density = counts(sub2ind(size(counts), binX, binY));

% Plot the scatter plot with color reflecting density
figure(); 
hh = scatter(x_lte, y_lte, 10, log(density), 'filled'); % Size of points set to 15, adjust as needed
set(gca,'ColorScale','log')
colorbar
xticks(0:100:max(x_lte))
yticks(0:100:max(y_lte))
colormap turbo;
hold on;
%plot(x_line, yfit, 'r-.', 'LineWidth', 1)
plot(x_line_lte, x_line_lte, 'r-', 'LineWidth', 2)
grid on
axis equal
xlabel('S_{meas}'); ylabel('S_{fit}');
title(sprintf('Correlation plot Kurtosis LTE, \\rho = %.6f',corr(x_lte,y_lte)))
xlim([-10 max(x_lte)+10])
ylim([-10 max(y_lte)+10])
toc;

%%% --- Linear regression ---
% P_lte = polyfit(S_meas_lte, S_fitv_lte, 1);
% 
% x_line_lte = linspace(min(S_meas_lte), max(S_meas_lte), 100);
% yfit_lte = polyval(P_lte, x_line_lte);

% Fit regression
% P_lte = polyfit(S_meas_lte_scat, S_fit_lte_scat, 1);
% 
% % Line for plotting
% x_line_lte = linspace(min(S_meas_lte_scat), max(S_meas_lte_scat), 100);
% yfit_lte = polyval(P_lte, x_line_lte);
% 
% scale_lte = max(S_meas_lte);
% 
% S_meas_norm_lte = S_meas_lte / scale_lte;
% S_fit_norm_lte  = S_fit_lte  / scale_lte;
% 
% few_points_lte = randperm(numel(S_meas_norm_lte), ...
%     min(50000,numel(S_meas_norm_lte)));
% 
% S_meas_norm_lte = S_meas_norm_lte(few_points_lte);
% S_fit_norm_lte  = S_fit_norm_lte(few_points_lte);

% figure;
% histogram(S_meas_norm_lte, 100, 'Normalization', 'probability')
% hold on
% histogram(S_fit_norm_lte, 100, 'Normalization', 'probability')
% legend('Measured','Fitted')
% title('Signal distributions (normalized)')
% xlabel('Signal value')
% ylabel('Probability')
% grid on


%%% STE %%%%
% S_measss_ste = data_lte_1D';   % now: voxels × gradients
% % S_fit_lte  = S_fit_lte;      % voxels × gradients
% x_ste = S_measss_ste(:);
% y_ste = S_fit_ste(:);
% data_slice_ste = squeeze(data_ste(:,:,slice,:));   % (x,y,g)
% S_fit_slice_ste  = squeeze(S_fit_ste(:,:,slice,:));  % (x,y,g)
% x_ste=data_slice_ste(:);
% y_ste=S_fit_slice_ste(:);



% --- LTE slice ---
data_slice_ste = squeeze(data_ste(:,:,slice,:));
fit_slice_ste  = squeeze(S_fit_ste_vol(:,:,slice,:));

x_ste = data_slice_ste(:);
y_ste = fit_slice_ste(:);

valid_ste = isfinite(x_ste) & isfinite(y_ste) & x_ste > 0 & y_ste > 0;
x_ste = x_ste(valid_ste);
y_ste = y_ste(valid_ste);


P_ste = polyfit(x_ste, y_ste, 1);

x_line_ste = linspace(min(x_ste), max(x_ste), 100);
yfit_ste = polyval(P_ste, x_line_ste);

% Define the grid for density calculation

tic;
% Define the grid for density calculation

numBins = 500; % Adjust number of bins based on data spread for smoother colors
[counts, ~, ~, binX, binY] = histcounts2(x_ste, y_ste, numBins);

% Map each point to its density
density = counts(sub2ind(size(counts), binX, binY));

% Plot the scatter plot with color reflecting density
figure(); 
h = scatter(x_ste, y_ste, 10, log(density), 'filled'); % Size of points set to 15, adjust as needed
set(gca,'ColorScale','log')
colorbar
xticks(0:100:max(x_ste))
yticks(0:100:max(y_ste))
colormap turbo;
hold on;
%plot(x_line, yfit, 'r-.', 'LineWidth', 1)
plot(x_line_ste, x_line_ste, 'r-', 'LineWidth', 2)
grid on
axis equal
xlabel('S_{meas}'); ylabel('S_{fit}');
title(sprintf('Correlation plot Kurtosis STE, \\rho = %.6f',corr(x_ste,y_ste)))
xlim([-10 max(x_ste)+10])
ylim([-10 max(y_ste)+10])
toc;


% %%% combined %%%%
% % Fit regression
% P_comb = polyfit(S_meas_comb_scat, S_fit_comb_scat, 1);
% 
% % Line for plotting
% x_line_comb = linspace(min(S_meas_comb_scat), max(S_meas_comb_scat), 100);
% yfit_comb = polyval(P_comb, x_line_comb);
% 
% scale_comb = max(S_meas_comb);
% 
% S_meas_norm_comb = S_meas_comb / scale_comb;
% S_fit_norm_comb  = S_fit_comb  / scale_comb;
% 
% few_points_comb = randperm(numel(S_meas_norm_comb), ...
%     min(50000,numel(S_meas_norm_comb)));
% 
% S_meas_norm_comb = S_meas_norm_comb(few_points_comb);
% S_fit_norm_comb  = S_fit_norm_comb(few_points_comb);
% 
% 
% figure;
% scatter_kde(S_meas_comb_scat, S_fit_comb_scat, ...
%     'MarkerSize', 5, 'filled')
% colorbar();
% hold on;
% plot(x_line_comb, yfit_comb, 'r-.', 'LineWidth', 0.5)
% axis equal
% grid on
% xlabel('Measured signal')
% ylabel('Fitted signal')
% title('Correlationsplot of STE+LTE data')
% 
% xlim([0 max(S_meas_comb_scat)])
% ylim([0 max(S_fit_comb_scat)])
% 
% figure;
% histogram(S_meas_norm_comb, 100, 'Normalization','probability')
% hold on
% histogram(S_fit_norm_comb, 100, 'Normalization','probability')
% legend('Measured','Fitted')
% title('Signal distributions (normalized)')
% xlabel('Signal value')
% ylabel('Probability')
% grid on
% 
% %%
% %%% Pearsons correlations coefficient
% R_lte = corrcoef(S_meas_lte_scat, S_fit_lte_scat);
% r_lte = R_lte(1,2);
% 
% R_ste = corrcoef(S_meas_ste_scat, S_fit_ste_scat);
% r_ste = R_ste(1,2);
% 
% R_comb = corrcoef(S_meas_comb_scat, S_fit_comb_scat);
% r_comb = R_comb(1,2);
% 
% disp(r_lte)
% disp(r_ste)
% disp(r_comb)
% 
