%%%%% Nonlinear least squares %%%%% 
clear all
clc
close all 

data = double(niftiread('LTE2.nii.gz'));
info = niftiinfo('LTE2.nii.gz');

bvals = load('LTE2.bvals'); bvals = bvals(:);
bvecs = load('LTE2.bvecs'); bvecs = bvecs';

N = length(bvals);

b_matrix = zeros(N, 6);
b_tensor = zeros(N, 15);
 
for i = 1:N
    bx = bvecs(i,1);
    by = bvecs(i,2);
    bz = bvecs(i,3);
    b  = bvals(i);

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

mask = bvals > 0;
X = [b_matrix, -1/6 * b_tensor];
X_mask = X(mask,:);
b0 = data(:,:,:,1);
th = prctile(b0(:), 90);        % robust high-intensity reference
bmask = b0 > 0.3 * th;

%%%%%% Output maps %%%%%%

[nx, ny, nz, ~] = size(data);

%%%%%% NLS model %%%%%%

function [F,J] = model_with_jac(p,X)
    e = exp(-X*p(2:22));
    F = p(1)*e;
    if nargout > 1
        J = zeros(size(X,1),22,'like',X);
        J(:,1) = e;
        J(:,2:22) = -p(1) * (e .* X);
    end
end

options = optimoptions('lsqcurvefit', ...
    'Display','off', ...
    'MaxIterations',20, ...
    'FunctionTolerance',1e-6, ...
    'StepTolerance',1e-6); ...
    'SpecifyObjectiveGradient';


data_1D = permute(data,[4,1,2,3]);
data_1D = data_1D(:,bmask);
data_1D(data_1D <= 0) = eps;

% nVox = nnz(bmask);
nVox = size(data_1D,2);
nb = length(bvals);

MD_map = zeros(nVox,1);
Dvals = zeros(nVox,3,3);
W_map  = zeros(nVox,1);
FA_map  = zeros(nVox,1);
Kpar_map = zeros(nVox,1);
Kperp_map = zeros(nVox,1);
S_fit = zeros(nVox, nb);   % 133 × nVox



lb = [0; -3*ones(6,1); -3*ones(15,1)];
ub = [Inf; 3*ones(6,1); 3*ones(15,1)];

exitflag = zeros(1,nVox); 
DW = cat(2,ones(N,1),-X) \ log(data_1D);
p0 = [exp(DW(1,:)); DW(2:end,:)];


%%%% Helper functions %%%%
function D = build_D_from_p(p)
    D = [p(2) p(5) p(6);
         p(5) p(3) p(7);
         p(6) p(7) p(4)];
end

function D = project_to_SPD(D)
    [V,L] = eig(D);
    L = diag(max(diag(L), 1e-10));  
    D = V * L * V';
end

function p = update_p_from_D(p, D)
    p(2) = D(1,1);
    p(3) = D(2,2);
    p(4) = D(3,3);
    p(5) = D(1,2);
    p(6) = D(1,3);
    p(7) = D(2,3);
end

for i = 1:nVox
    D0 = build_D_from_p(p0(:,i));
    D0 = project_to_SPD(D0);
    p0(:,i) = update_p_from_D(p0(:,i), D0);
end

function Kdir = directional_kurtosis(W, n)

nx = n(1);
ny = n(2);
nz = n(3);

Kdir = ...
    W(1)*nx^4 + W(2)*ny^4 + W(3)*nz^4 ...
    + W(4)*nx^3*ny + W(5)*ny^3*nx ...
    + W(6)*nx^3*nz + W(7)*nz^3*nx ...
    + W(8)*ny^3*nz + W(9)*nz^3*ny ...
    + W(10)*nx^2*ny^2 ...
    + W(11)*nx^2*nz^2 ...
    + W(12)*ny^2*nz^2 ...
    + W(13)*nx^2*ny*nz ...
    + W(14)*ny^2*nx*nz ...
    + W(15)*nz^2*ny*nx;
end


%%%% Main loop %%%%
parfor idx=1:nVox

    S = data_1D(:,idx); 

    % Avoid log of zero
    S(S <= 0) = eps;


    [p,~,~,exitflag] = lsqcurvefit(@(p,X)model_with_jac(p,X), p0(:,idx), X, S, lb, ub, options);
    
    %%% Save fitted signal %%%
    S_fit(idx,:) = model_with_jac(p, X);

    if exitflag <= 0 || any(~isfinite(p))
        continue
    end
    
    % --- Build D ---
    D = build_D_from_p(p);
    D = project_to_SPD(D);
    p = update_p_from_D(p, D);
    
    % --- Metrics ---
    % lambda = eig(D);
    [V,L] = eig(D);

    lambda = diag(L);
    [lambda, order] = sort(lambda,'descend');
    
    V = V(:,order);


    MD = mean(lambda);
    Dpar = lambda(1);
    Dperp = (lambda(2) + lambda(3))/2;

    FA = sqrt(3/2 * sum((lambda - MD).^2) / (sum(lambda.^2)+eps));
    
    MD_map(idx) = MD;

    
    W = p(8:22);
    W = W/(MD^2);
    
    n1 = V(:,1);
    Wpar = directional_kurtosis(W, n1);
    
    % Kpar1 = directional_kurtosis(W, n1);
    % Kpar2 = directional_kurtosis(W, -n1);
    % 
    % Kpar_map(idx) = mean([Kpar1, Kpar2]) / (Dpar^2 + eps);
    if Dpar > 0
        Kpar_map(idx) = Wpar * (MD^2) / (Dpar^2);
    end
    n2 = V(:,2);
    n3 = V(:,3);
    
    Wperp2 = directional_kurtosis(W, n2);
    Wperp3 = directional_kurtosis(W, n3);
    
    Wperp = (Wperp2 + Wperp3)/2;
    if Dperp < 0.1e-3
        continue
    end
    
    if Dperp > 0
        Kperp_map(idx) = Wperp * (MD^2) / (Dperp^2);
    end

    W_map(idx) = sum(W([1:3,10:12]))/5;
    
    FA_map(idx) = FA;

end


%%%%%% Visualization %%%%%%
MD_vol = zeros(nx, ny, nz);
FA_vol = zeros(nx, ny, nz);
W_vol  = zeros(nx, ny, nz);
Kpar_vol  = zeros(nx, ny, nz);
Kperp_vol = zeros(nx, ny, nz);

MD_vol(bmask) = MD_map;
FA_vol(bmask) = FA_map;
W_vol(bmask)  = W_map;
Kpar_vol(bmask)  = Kpar_map;
Kperp_vol(bmask) = Kperp_map;

%save("DKI_NLS.mat", "MD_vol", "FA_vol", "W_vol", "Kpar_vol", "Kperp_vol")

% info.Filename = 'FA_output.nii.gz';
% 
% niftiwrite(FA_vol, 'FA_output', info, 'Compressed', true);

slice = round(nz/2);

MD_plot = MD_vol(:,:,slice);
FA_plot = FA_vol(:,:,slice);
W_plot  = W_vol(:,:,slice);
Kpar_plot  = Kpar_vol(:,:,slice);
Kperp_plot = Kperp_vol(:,:,slice);


%%%%%% Plot MD %%%%%%

figure;
imagesc(rot90(MD_plot,-1));
axis image off;
colormap turbo;
colorbar;
title(['Mean Diffusivity (MD) - Slice ', num2str(slice)]);


%%%%%% Plot FA %%%%%%

figure;
imagesc(rot90(FA_plot,-1), [0 1]);
axis image off;
colormap turbo;
colorbar;
title(['Fractional Anisotropy (FA) - Slice ', num2str(slice)]);

%%%%%% Plot W %%%%%%

figure;
imagesc(rot90(W_plot,-1), [0 2]);
axis image off;
colormap turbo;
colorbar;
title(['Mean Kurtosis (MK) - Slice ', num2str(slice)])

figure;
imagesc(rot90(Kpar_plot,-1), [0 2]);
axis image off;
colormap turbo;
colorbar;
title(['K parallel - Slice ', num2str(slice)])

figure;
imagesc(rot90(Kperp_plot,-1), [0 2]);
axis image off;
colormap turbo;
colorbar;
title(['K perpendicular - Slice ', num2str(slice)])
%%
b0 = mean(data(:,:,:,bvals==0),4);
mask = b0 > 0.1*max(b0(:));

% WM_mask = FA_vol > 0.3;

WM_mask = ...
    (FA_vol > 0.5) & ...
    (MD_vol > 0.5e-3) & ...
    (MD_vol < 1.1e-3);
% 
% WM_mask = bwmorph3(WM_mask,'erode',1);

FA_vals = FA_vol(WM_mask);
MD_vals = MD_vol(WM_mask);
W_vals = W_vol(WM_mask);
Kpar_vals = Kpar_vol(WM_mask);
Kper_vals = Kperp_vol(WM_mask);



figure;
histogram(FA_vals, 100, 'Normalization', 'probability');
xlabel('FA');
ylabel('Voxel count');
title('FA distribution WM mask');
figure;
histogram(FA_vol(mask), 100, 'Normalization', 'probability');
xlabel('FA');
ylabel('Voxel count');
title('FA distribution mask');

figure;
histogram(MD_vals, 100, 'Normalization', 'probability');
xlabel('MD (mm^2/s)');
ylabel('Voxel count');
title('MD distribution');

figure;
histogram(W_vals, 100, 'Normalization', 'probability');
xlabel('W tot');
ylabel('Voxel count');
title('W tot distribution');

figure;
histogram(Kpar_vals, 100, 'Normalization', 'probability');
xlabel('K par');
ylabel('Voxel count');
title('K parallel distribution');

figure;
histogram(Kper_vals, 100, 'Normalization', 'probability');
xlabel('K per');
ylabel('Voxel count');
title('K perpendicular distribution');

%%

% S_meas_mat = data_1D';   % now: voxels × gradients
% S_fit_mat  = S_fit;      % voxels × gradients
% x = S_meas_mat(:);
% y = S_fit_mat(:);

slice = round(size(data,3)/2);
%data_slice=data_1D';
data_slice = squeeze(data(:,:,slice,:));   % (x,y,g)
S_fit_slice  = squeeze(S_fit(:,:,slice,:));  % (x,y,g)
x=data_slice(:);
y=S_fit_slice(:);

% Define the grid for density calculation
P = polyfit(x, y, 1);

x_line = linspace(min(x), max(x), 100);
yfit = polyval(P, x_line);

tic;
% Define the grid for density calculation

numBins = 500; % Adjust number of bins based on data spread for smoother colors
[counts, ~, ~, binX, binY] = histcounts2(x, y, numBins);

% Map each point to its density
density = counts(sub2ind(size(counts), binX, binY));

% Plot the scatter plot with color reflecting density
figure(); 
hh = scatter(x, y, 10, log(density), 'filled'); % Size of points set to 15, adjust as needed
set(gca,'ColorScale','log')
colorbar
xticks(0:100:max(x))
yticks(0:100:max(y))
colormap turbo;
hold on;
%plot(x_line, yfit, 'r-.', 'LineWidth', 1)
plot(x_line, x_line, 'r-', 'LineWidth', 2)
grid on
axis equal
xlabel('S_{meas}'); ylabel('S_{fit}');
title(sprintf('Correlation plot DKI NLS, \\rho = %.6f',corr(x,y)))
xlim([-10 max(x)+10])
ylim([-10 max(y)+10])
toc;

% 
% figure;
% histogram(S_meas_norm, 100, 'Normalization', 'probability')
% hold on
% histogram(S_fit_norm, 100, 'Normalization', 'probability')
% legend('Measured','Fitted')
% title('Signal distributions (normalized)')
% xlabel('Normalized signal value')
% ylabel('Probability')
% grid on
% 
% 
% R_lte = corrcoef(S_meas, S_fitv);
% r_lte = R_lte(1,2);
% 
% disp(r_lte)
% 
