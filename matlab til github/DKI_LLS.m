%Linear least squares. Finding W's.
clear all
clc
close all 

data = niftiread('LTE2.nii.gz');
info = niftiinfo('LTE2.nii.gz');

bvals = load('LTE2.bvals');
bvecs = load('LTE2.bvecs');
bvals=bvals';
bvecs=bvecs';


b_matrix=zeros(65, 6);
b_tensor=zeros(65, 15);
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
    
    b_tensor(i, :)=[
        b^2*bx*bx*bx*bx,
        b^2*by*by*by*by,
        b^2*bz*bz*bz*bz,
        4*b^2*bx*bx*bx*by,
        4*b^2*by*by*by*bx,
        4*b^2*bx*bx*bx*bz, 
        4*b^2*bz*bz*bz*bx,
        4*b^2*by*by*by*bz,
        4*b^2*bz*bz*bz*by,
        6*b^2*bx*bx*by*by,
        6*b^2*bx*bx*bz*bz,
        6*b^2*by*by*bz*bz,
        12*b^2*bx*bx*by*bz,
        12*b^2*by*by*bx*bz,
        12*b^2*bz*bz*by*bx];

end

[nx, ny, nz, n]=size(data);
MD_map=zeros(nx, ny, nz);
FA_map=zeros(nx, ny, nz);
evec_map=zeros(nx, ny, nz, 3, 3);
W_d=zeros(nx, ny, nz);
W_parallel=zeros(nx, ny, nz);
W_perpen=zeros(nx, ny, nz);
K_parallel=zeros(nx, ny, nz);
K_perpen=zeros(nx, ny, nz);

S_fit=zeros(nx, ny, nz, n);
S_meas=zeros(nx, ny, nz, n);


mask=bvals>0;
b_mask=b_matrix(mask, :);
X=[b_matrix -1/6*b_tensor];
X_mask=X(mask, :);

for x=1:nx
    for y = 1:ny
        for z = 1:nz
            S = double(squeeze(data(x, y, z, :))); 
            S0 = mean(S(~mask));

            if any(S(mask) <= 0)
                continue
            end

            if S0<20
                continue
            end

            Y=-log(S/S0);
            Y_fit=Y(mask);
            dw=X_mask\Y_fit;

            % Save signal for korrelation plot
            Y_fit_model = X * dw;
            S_fit(x, y, z, :) = S0 * exp(-Y_fit_model);
            S_meas(x,y,z,:)=S;


            %Extract elements
            d_elem=dw(1:6);
            w_elem=dw(7:21);

            D=[dw(1) dw(4) dw(5);
               dw(4) dw(2) dw(6);
               dw(5) dw(6) dw(3)];

            [V, L] = eig(D);
            V=real(V);
            L=real(L);

            evals=diag(L);
            [evals, idx]=sort(evals, 'descend');
            evecs=V(:, idx);

            evec_map(x,y,z, :, :)=evecs;
            
            MD_val=mean(evals); %MD value to get real W
            D_parallel = evals(1);
            D_perp = (evals(2) + evals(3))/2;
            W=w_elem/MD_val^2; %W without avg(D)^2 

            trW=W(1)+W(2)+W(3)+2*W(10)+2*W(11)+2*W(12); %Trace(W)
            avg_W=1/5*trW;
            W_d(x, y, z)=avg_W;

            %%%%%% Computing Parallel W %%%%%%
            primary_n=evecs(:,1);
            nnx=primary_n(1);
            nny=primary_n(2);
            nnz=primary_n(3);
            par=W(1)*nnx^4+W(2)*nny^4+W(3)*nnz^4 ...
                +W(4)*nnx^3*nny+W(5)*nny^3*nnx ...
                +W(6)*nnx^3*nnz+W(7)*nnz^3*nnx ...
                +W(8)*nny^3*nnz+W(9)*nnz^3*nny ...
                +W(10)*nnx^2*nny^2 ...
                +W(11)*nnx^2*nnz^2 ...
                +W(12)*nny^2*nnz^2 ...
                +W(13)*nnx^2*nny*nnz ...
                +W(14)*nny^2*nnx*nnz ...
                +W(15)*nnz^2*nny*nnx;

            W_parallel(x,y,z)=par;

            if D_parallel > 0
                K_parallel(x,y,z) = par * (MD_val^2) / (D_parallel^2);
            end

            %%%% Computing Perpendicular W %%%%%

            n2=evecs(:,2);
            n3=evecs(:,3);
            n4=(n2+n3)/sqrt(2);
            n5=(n2-n3)/sqrt(2);

            % n2 = n2 / norm(n2);
            % n3 = n3 / norm(n3);
            % n4 = n4 / norm(n4);
            % n5 = n5 / norm(n5);

            n2nx=n2(1);
            n2ny=n2(2);
            n2nz=n2(3);
            n3nx=n3(1);
            n3ny=n3(2);
            n3nz=n3(3);
            n4x=n4(1);
            n4y=n4(2);
            n4z=n4(3);
            n5x=n5(1);
            n5y=n5(2);
            n5z=n5(3);


            per2=W(1)*n2nx^4+W(2)*n2ny^4+W(3)*n2nz^4 ...
                +W(4)*n2nx^3*n2ny+W(5)*n2ny^3*n2nx ...
                +W(6)*n2nx^3*n2nz+W(7)*n2nz^3*n2nx ...
                +W(8)*n2ny^3*n2nz+W(9)*n2nz^3*n2ny ...
                +W(10)*n2nx^2*n2ny^2 ...
                +W(11)*n2nx^2*n2nz^2 ...
                +W(12)*n2ny^2*n2nz^2 ...
                +W(13)*n2nx^2*n2ny*n2nz ...
                +W(14)*n2ny^2*n2nx*n2nz ...
                +W(15)*n2nz^2*n2ny*n2nx;

            per3=W(1)*n3nx^4+W(2)*n3ny^4+W(3)*n3nz^4 ...
                +W(4)*n3nx^3*n3ny+W(5)*n3ny^3*n3nx ...
                +W(6)*n3nx^3*n3nz+W(7)*n3nz^3*n3nx ...
                +W(8)*n3ny^3*n3nz+W(9)*n3nz^3*n3ny ...
                +W(10)*n3nx^2*n3ny^2 ...
                +W(11)*n3nx^2*n3nz^2 ...
                +W(12)*n3ny^2*n3nz^2 ...
                +W(13)*n3nx^2*n3ny*n3nz ...
                +W(14)*n3ny^2*n3nx*n3nz ...
                +W(15)*n3nz^2*n3ny*n3nx;

            per4=W(1)*n4x^4+W(2)*n4y^4+W(3)*n4z^4 ...
                +W(4)*n4x^3*n4y+W(5)*n4y^3*n4x ...
                +W(6)*n4x^3*n4z+W(7)*n4z^3*n4x ...
                +W(8)*n4y^3*n4z+W(9)*n4z^3*n4y ...
                +W(10)*n4x^2*n4y^2 ...
                +W(11)*n4x^2*n4z^2 ...
                +W(12)*n4y^2*n4z^2 ...
                +W(13)*n4x^2*n4y*n4z ...
                +W(14)*n4y^2*n4x*n4z ...
                +W(15)*n4z^2*n4y*n4x;

            per5=W(1)*n5x^4+W(2)*n5y^4+W(3)*n5z^4 ...
                +W(4)*n5x^3*n5y+W(5)*n5y^3*n5x ...
                +W(6)*n5x^3*n5z+W(7)*n5z^3*n5x ...
                +W(8)*n5y^3*n5z+W(9)*n5z^3*n5y ...
                +W(10)*n5x^2*n5y^2 ...
                +W(11)*n5x^2*n5z^2 ...
                +W(12)*n5y^2*n5z^2 ...
                +W(13)*n5x^2*n5y*n5z ...
                +W(14)*n5y^2*n5x*n5z ...
                +W(15)*n5z^2*n5y*n5x;
        
            if D_perp < 0.1e-3
                continue
            end
            W_perpen(x,y,z)=1/4*(per2+per3+per4+per5);
            if D_perp > 0
                K_perpen(x,y,z) = W_perpen(x,y,z) * (MD_val^2) / (D_perp^2);
            end

            MD_map(x,y,z) = mean(evals);


            FA_map(x,y,z)=sqrt(3/2*((evals(1)-MD_val)^2+(evals(2)-MD_val)^2+(evals(3)-MD_val)^2) ...
                /(evals(1)^2+evals(2)^2+evals(3)^2));
        end
    end
end


slice=round(size(data, 3)/2);

% quiver plot start
step=1;

FA_end=size(FA_map);
x=1:step:FA_end(2);
y=1:step:FA_end(1);

[X, Y]= meshgrid(x, y);
Xr = rot90(X, 1);
Yr = rot90(Y, 1);

U = squeeze(evec_map(1:step:end, 1:step:end, slice, 1, 1));
V = squeeze(evec_map(1:step:end, 1:step:end, slice, 2, 1));

% rotate vector field 90° clockwise
U_rot = -V;
V_rot = U;

% Display a slice of the Mean Diffusivity (MD) map

figure;
imagesc(rot90(W_d(:,:,slice),-1), [0 2]);
colormap turbo;
colorbar;
title(['Mean kurtosis (MK) - Slice ', num2str(slice)]);
axis image off;

figure;
imagesc(rot90(W_parallel(:,:,slice),-1), [0 2]);
colormap turbo;
colorbar;
title(['W parallel - Slice ', num2str(slice)]);
axis image off;

figure;
imagesc(rot90(W_perpen(:,:,slice),-1), [0 2]);
colormap turbo;
colorbar;
title(['W perpendicular - Slice ', num2str(slice)]);
axis image off;

figure;
imagesc(rot90(K_parallel(:,:,slice),-1), [0 2]);
colormap turbo;
colorbar;
title(['K parallel - Slice ', num2str(slice)]);
axis image off;

figure;
imagesc(rot90(K_perpen(:,:,slice),-1), [0 2]);
colormap turbo;
colorbar;
title(['K perpendicular - Slice ', num2str(slice)]);
axis image off;


figure;
imagesc(rot90(MD_map(:,:,slice), -1));
colormap turbo;
colorbar;
title(['Mean Diffusivity Map - Slice ', num2str(slice)]);
axis image off;

figure;
imagesc(rot90(FA_map(:,:,slice),-1), [0 1]);
colormap turbo;
colorbar;
title(['Fractional anisotrope - Slice ', num2str(slice)]);
axis image off;

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

%%Histogram analyses
b0 = mean(data(:,:,:,bvals==0),4);
mask = b0 > 0.2*max(b0(:));

% WM_mask = FA_map > 0.3;
WM_mask = ...
    (FA_map > 0.5) & ...
    (MD_map > 0.5e-3) & ...
    (MD_map < 1.1e-3);

FA_vals = FA_map(WM_mask);
MD_vals = MD_map(WM_mask);
W_vals = W_d(WM_mask);
Kpar_vals = K_parallel(WM_mask);
Kper_vals = K_perpen(WM_mask);



figure;
histogram(FA_vals, 100, 'Normalization', 'probability');
xlabel('FA');
ylabel('Voxel count');
title('FA distribution');

figure;
histogram(MD_vals, 100, 'Normalization', 'probability');
xlabel('MD (mm^2/s)');
ylabel('Voxel count');
title('MD distribution');

figure;
histogram(W_vals, 100, 'Normalization', 'pdf');
xlabel('W tot');
ylabel('Voxel count');
title('W tot distribution');
 
figure;
histogram(Kpar_vals, 100, 'Normalization', 'pdf');
xlabel('K par');
ylabel('Voxel count');
title('K parallel distribution');
 
figure;
histogram(Kper_vals, 100, 'Normalization', 'pdf');
xlabel('K per');
ylabel('Voxel count');
title('K perpendicular distribution');


%%
%%% Korrelation plot %%%

data_slice = squeeze(data(:,:,slice,:));   % (x,y,g)
S_fit_slice  = squeeze(S_fit(:,:,slice,:));  % (x,y,g)
x=data_slice(:);
y=S_fit_slice(:);


%%% --- Linear regression ---
P = polyfit(x, y, 1);

x_line = linspace(min(x), max(x), 100);
yfit = polyval(P, x_line);

% Define the grid for density calculation
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
title(sprintf('Correlation plot DKI LLS, \\rho = %.6f',corr(x,y)))
xlim([-10 max(x)+10])
ylim([-10 max(y)+10])
