%%%%%%% DTI %%%%%%%%%%%

clear all
clc
close all 

set(groot, ...
    'DefaultAxesFontSize', 16, ...
    'DefaultTextFontSize', 16, ...
    'DefaultLegendFontSize', 14);
set(groot, 'defaultFigureRenderer', 'painters')

data = niftiread('LTE2.nii.gz');
info = niftiinfo('LTE2.nii.gz');

bvals = load('LTE2.bvals');
bvecs = load('LTE2.bvecs');
bvals=bvals';
bvecs=bvecs';


b_matrix=zeros(65, 6);
for i = 1:length(bvals)

    bx=bvecs(i,1);
    by=bvecs(i,2);
    bz=bvecs(i,3);

    b=bvals(i);
    b_matrix(i, :) = [
        b*bx*bx,
        b*by*by,
        b*bz*bz,
        2*b*bx*by,
        2*b*bx*bz,
        2*b*by*bz];
end


[nx, ny, nz, n]=size(data);
MD_map=zeros(nx, ny, nz);
FA_map=zeros(nx, ny, nz);
evec_map=zeros(nx, ny, nz, 3, 3);
% S_fit=zeros(nx, ny, nz, n);
% S_meas=zeros(nx, ny, nz, n);
S_fit=zeros(nx, ny, nz, n);



mask=bvals>0;
b_mask=b_matrix(mask, :);
b0_vol = mean(data(:,:,:,bvals==0), 4);

for x=1:nx
    for y = 1:ny
        for z = 1:nz

            S = double(squeeze(data(x, y, z, :))); 

            S0 = b0_vol(x,y,z);

            if any(S(mask) <= 0)
                continue
            end

            if S0<20
                continue
            end

            Y=-log(S/S0);
            Y_fit=Y(mask);
            %Y = -log(max(S/S0, 1e-10));
            d=b_mask\Y_fit;

 
            % korre
            Y_pred = b_matrix * d;
            S_fit(x, y, z, :) = S0 * exp(-Y_pred);


            D=[d(1) d(4) d(5);
               d(4) d(2) d(6);
               d(5) d(6) d(3)];


            [V, L] = eig(D);
            V=real(V);
            L=real(L);

            evals=diag(L);
            [evals, idx]=sort(evals, 'descend');
            evecs=V(:, idx);

            evec_map(x,y,z, :, :)=evecs;

            %ev=sort(eigvals, 'descend');
            

            MD_map(x,y,z) = mean(evals);
            MD_val=mean(evals);

            FA_map(x,y,z)=sqrt(3/2*((evals(1)-MD_val)^2+(evals(2)-MD_val)^2+(evals(3)-MD_val)^2) ...
                /(evals(1)^2+evals(2)^2+evals(3)^2));



        end
    end
end

%slice=30;
slice=round(size(data, 3)/2);

%quiver plot start
step=1;

FA_end=size(FA_map);
x=1:step:FA_end(2);
y=1:step:FA_end(1);

[X, Y]= meshgrid(x, y);
Xr = rot90(X, 1);
Yr = rot90(Y, 1);

U = squeeze(evec_map(1:step:end, 1:step:end, slice, 1, 1));
V = squeeze(evec_map(1:step:end, 1:step:end, slice, 2, 1));
U_rot = -V;
V_rot = U;

% Display a slice of the Mean Diffusivity (MD) map

%subplot(2,2, 1);
figure;
imagesc(rot90(MD_map(:,:,slice),-1), [0 0.0045]);
colormap turbo;
colorbar;
title(['Mean Diffusivity Map - Slice ', num2str(slice)]);
axis image off;

figure;
imagesc(rot90(FA_map(:,:,slice),-1), [0 1]);
colormap gray;
colorbar;
title(['Fractional anisotrope - Slice ', num2str(slice)]);
axis image off;

%subplot(2,2,2);
figure;
imagesc(rot90(FA_map(:,:,slice),-1), [0 1]);
colormap gray;
colorbar;
title(['Fractional anisotrope - Slice ', num2str(slice)]);
axis image off;
hold on

q=quiver(Xr, Yr, V_rot, -U_rot, 0, 'r', 'linewidth',1);
q.ShowArrowHead = 'off';
q.Marker = 'none';

% b0 = mean(data(:,:,:,bvals==0),4);

b0 = mean(data(:,:,:,bvals==0),4);
mask = b0 > 0.2*max(b0(:));

% csf_mask = b0 < prctile(b0(mask), 10);
% brain_mask_no_csf = mask & ~csf_mask;
WM_mask = FA_map > 0.3;

FA_vals = FA_map(mask);
MD_vals = MD_map(mask);
%%
WM_mask = FA_map > 0.3;
figure;
histogram(FA_vals, 100, 'Normalization', 'probability');
xlabel('FA');
ylabel('Voxel count');
title('FA distribution with white matter mask');
%%
figure;
histogram(MD_vals, 100, 'Normalization', 'pdf');
xlabel('MD (mm^2/s)');
ylabel('Voxel count');
title('MD distribution');



%%

clear all
S=load('DTI_correlation_slice.mat');
x=S.x(:);
y=S.y(:);

%%% --- Linear regression ---
P = polyfit(x, y, 1);

x_line = linspace(min(x), max(x), 100);
yfit = polyval(P, x_line);


numBins = 500; % Adjust number of bins based on data spread for smoother colors
[counts, ~, ~, binX, binY] = histcounts2(x, y, numBins);

% Map each point to its density
density = counts(sub2ind(size(counts), binX, binY));

% Plot the scatter plot with color reflecting density
figure(); 
hh = scatter(x, y, 15, log(density), 'filled'); % Size of points set to 15, adjust as needed
set(gca,'ColorScale','log')
colorbar
xticks(0:100:max(x))
yticks(0:100:max(y))
colormap parula;
hold on;
%plot(x_line, yfit, 'r-.', 'LineWidth', 1)
plot(x_line, x_line, 'r-', 'LineWidth', 2)
grid on
axis equal
xlabel('S_{meas}'); ylabel('S_{fit}');
title(sprintf('Correlation plot DTI, \\rho = %.6f',corr(x,y)))
xlim([-10 max(x)+10])
ylim([-10 max(y)+10])

residual=x-y;

% 95% interval
lims = prctile(residual,[2.5 97.5]);

figure;
histogram(residual,300);
xlim(lims)
xlabel('Residual (S_{meas} - S_{fit})');
ylabel('Count');
title('Residual distribution');
grid on;


% Nboot = 1000;
% n = length(x);
% rboot = zeros(Nboot,1);
% 
% for i = 1:Nboot
%     idx = randi(n, n, 1);
%     rboot(i) = corr(x(idx), y(idx));
% end
% 
% CI = prctile(rboot, [2.5 97.5]);
% 
% r = corr(x,y);
% 
% fprintf('r = %.4f (95%% CI: %.4f – %.4f)\n', r, CI(1), CI(2));
% SS_res = sum((x - y).^2);
% 
% SS_tot = sum((x - mean(x)).^2);
% 
% R2 = 1 - SS_res/SS_tot;


% legend('Data density', 'Linear fit', 'y = x', 'Location', 'best')

% idx = x > 300;
% c = log(density);
% figure();
% h = scatter(x, y, 15, c, 'filled');
% colormap turbo
% drawnow
% 
% fprintf('nnz(x>300) original data:     %d\n', nnz(idx))
% fprintf('nnz(h.XData>300) scatter obj: %d\n', nnz(h.XData(:)>300))
% fprintf('numel(h.XData):              %d\n', numel(h.XData))
% fprintf('numel(x):                    %d\n', numel(x))
% fprintf('finite CData x>300:           %d\n', nnz(idx & isfinite(c)))
% fprintf('unique CData x>300:           %d\n', numel(unique(c(idx))))
% figure;
% scatter(x(x>300), y(x>300), 20, 'filled')
% slice = round(size(data,3)/2);
% data_slice = squeeze(data(:,:,slice,:));   % (x,y,g)
% S_fit_slice  = squeeze(S_fit(:,:,slice,:));  % (x,y,g)
% x=data_slice(:);
% y=S_fit_slice(:);
% 
% % Define the grid for density calculation
% 
% numBins = 500; % Adjust number of bins based on data spread for smoother colors
% [counts, ~, ~, binX, binY] = histcounts2(x, y, numBins);
% 
% % Map each point to its density
% density = counts(sub2ind(size(counts), binX, binY));
% 
% % Plot the scatter plot with color reflecting density
% figure(); 
% h = scatter(x, y, 15, log(density), 'filled'); % Size of points set to 15, adjust as needed
% colormap turbo
% xlabel('S_{meas}'); ylabel('S_{fit}');
% title(sprintf('rho = %.6f',corr(x,y)))
%%