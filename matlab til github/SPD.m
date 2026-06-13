function D = SPD(D)
    [V,L] = eig(D);
    L = diag(max(diag(L), 1e-10));
    D = V*L*V';
end