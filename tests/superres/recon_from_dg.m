clear all;
close all;

params = {};
count = 0;
for z_plus = [650, 700, 750, 800, 850]
    for stage = ["dg", "mlp_pred", "cpmlp_pred"]
        for Nxdg = [8, 16, 24, 32, 48, 64]
            count = count+1;
            params{count} = {z_plus, stage, Nxdg};
        end
    end
end

parfor i = 1:length(params)
    par = params{i};
    reconstruct_dg(par{1}, par{2}, par{3})
end

function reconstruct_dg(z_plus, stage, Nxdg)
    truth_fn = "data/snap"+z_plus+".mat";
    load(truth_fn)
    dg_fn = "data/"+stage+z_plus+"_"+Nxdg+".mat";
    load(dg_fn)

    if stage=="dg"
        output_NN = dg_data(:,38:end);
    else
        output_NN = dg_data;
    end
    
    Nzdg = Nxdg/2;
    p_DG = 1;

    %Element info generate
    dx_dg = Lx / Nxdg;
    dz_dg = Lz / Nzdg;
    x_ref = linspace(0 + dx_dg / 2, Lx - dx_dg / 2, Nxdg);
    z_ref = linspace(0 + dz_dg / 2, Lz - dz_dg / 2, Nzdg);
    Nx_interp = 2 * ceil(dx_dg / (2 * dx)) + 1; %done to make sure odd intergation point (even interval)
    Nz_interp = 2 * ceil(dz_dg / (2 * dz)) + 1;
    x_interp = linspace(-1, 1, Nx_interp) * dx_dg / 2;
    z_interp = linspace(-1, 1, Nz_interp) * dz_dg / 2;
    x_interpr = linspace(-1, 1, Nx_interp);
    z_interpr = linspace(-1, 1, Nz_interp);

    Basis_eval = zeros(Nx_interp, Nz_interp, (p_DG + 1)^2);

    %Evalaute all basis at all these points
    for i = 1:(p_DG + 1)

        for j = 1:(p_DG + 1)
            nindex = (j - 1) * (p_DG + 1) + i;

            for i1 = 1:Nx_interp

                for j1 = 1:Nz_interp
                    Basis_eval(i1, j1, nindex) = eval_lag2D(x_interpr(i1), z_interpr(j1), p_DG, i, j);
                end

            end

        end

    end

    %2nd order Numerical Integration matrix
    numint = ones(Nx_interp, Nz_interp);
    numint(1, :) = 1/2;
    numint(Nx_interp, :) = 1/2;
    numint(:, 1) = 1/2;
    numint(:, Nz_interp) = 1/2;
    numint(1, 1) = 1/4;
    numint(Nx_interp, Nz_interp) = 1/4;
    numint(1, Nz_interp) = 1/4;
    numint(Nx_interp, 1) = 1/4;
    dx_dg2 = dx_dg / (Nx_interp - 1);
    dz_dg2 = dz_dg / (Nz_interp - 1);
    numint = numint * dx_dg2 * dz_dg2;

    %Store these points in a matrix for interpolation
    [Zq2, Xq2] = meshgrid(z_interp, x_interp);

    %Create p polynomial mass matrix
    [Zqr, Xqr] = meshgrid(z_interp, x_interp);

    %%Make Mass Matrix
    %Get Gauss Quadrature Points and weights
    [pos, weights] = eval_quad2D(p_DG);
    basis_qd = zeros((p_DG + 1)^2, size(pos, 1));

    %Compute basis function at quadrature points
    for i = 1:(p_DG + 1)

        for j = 1:(p_DG + 1)
            index = (j - 1) * (p_DG + 1) + i;
            basis_qd(index, :) = eval_lag2D(pos(:, 1), pos(:, 2), p_DG, i, j);
        end

    end

    size_MM = size(basis_qd, 1);
    Jac = (dx_dg / 2) * (dz_dg / 2);

    %Assemble mass matrix
    for i = 1:size_MM

        for j = 1:size_MM
            MassMat(i, j) = sum(basis_qd(i, :) .* basis_qd(j, :) .* weights) * Jac;
        end

    end

    inv_MM = inv(MassMat);

    %Save coefficents in matrix
    L2_solU = zeros(Nyplus, Nxdg, Nzdg, size_MM);
    L2_solV = zeros(Nyplus, Nxdg, Nzdg, size_MM);
    L2_solW = zeros(Nyplus, Nxdg, Nzdg, size_MM);
    %%

    hack = 0;

    %L2 Project DNS data
    for ypi = 1:Nyplus
        yindex = yplusi(ypi);
        Uplane = squeeze(Usnap(:, :, ypi));
        Vplane = squeeze(Vsnap(:, :, ypi));
        Wplane = squeeze(Wsnap(:, :, ypi));
        Uplane = extend(Uplane);
        Vplane = extend(Vplane);
        Wplane = extend(Wplane);

        zp = linspace(0, pi, nz + 1);
        xp = linspace(0, 2 * pi, nx + 1);

        [Z2d, X2d] = meshgrid(zp, xp);

        %HACK VARIABLE
        if (hack == 1)
            Uplane = X2d + 2 * Z2d;
        end

        %Element loop
        for i = 1:Nxdg

            for j = 1:Nzdg
                x_interp2 = x_interp + x_ref(i);
                z_interp2 = z_interp + z_ref(j);
                [Zq, Xq] = meshgrid(z_interp2, x_interp2);

                Uq = interp2(zp, xp, Uplane, Zq, Xq, 'cubic');
                Vq = interp2(zp, xp, Vplane, Zq, Xq, 'cubic');
                Wq = interp2(zp, xp, Wplane, Zq, Xq, 'cubic');

                RUq2 = zeros(size_MM, 1);
                RVq2 = zeros(size_MM, 1);
                RWq2 = zeros(size_MM, 1);

                for k = 1:size_MM
                    RUq2(k) = simpson2D(Uq .* Basis_eval(:, :, k), dx_dg2, dz_dg2);
                    RVq2(k) = simpson2D(Vq .* Basis_eval(:, :, k), dx_dg2, dz_dg2);
                    RWq2(k) = simpson2D(Wq .* Basis_eval(:, :, k), dx_dg2, dz_dg2);
                end

                nodeU = MassMat \ RUq2;
                nodeV = MassMat \ RVq2;
                nodeW = MassMat \ RWq2;

                L2_solU(ypi, i, j, :) = nodeU(:);
                L2_solV(ypi, i, j, :) = nodeV(:);
                L2_solW(ypi, i, j, :) = nodeW(:);

            end

        end

    end

    %get N=4 solution back
    p_DG2 = 3;

    %Create new Mass Matrix
    %Get Gauss Quadrature Points and weights
    [pos2, weights2] = eval_quad2D(p_DG2);
    basis_qd2 = zeros((p_DG2 + 1)^2, size(pos2, 1));

    %Compute basis function at quadrature points
    for i = 1:(p_DG2 + 1)

        for j = 1:(p_DG2 + 1)
            index = (j - 1) * (p_DG2 + 1) + i;
            basis_qd2(index, :) = eval_lag2D(pos2(:, 1), pos2(:, 2), p_DG2, i, j);
        end

    end

    size_MM2 = size(basis_qd2, 1);
    Jac2 = (dx_dg / 2) * (dz_dg / 2);
    MassMat2 = zeros(size_MM2, size_MM2);

    %Assemble mass matrix
    for i = 1:size_MM2

        for j = 1:size_MM2
            MassMat2(i, j) = sum(basis_qd2(i, :) .* basis_qd2(j, :) .* weights2) * Jac2;
        end

    end

    inv_MM2 = inv(MassMat2);
    %Save coefficents in matrix
    L2_solU2 = zeros(Nyplus, Nxdg, Nzdg, size_MM2);
    L2_solU2_L = zeros(Nyplus, Nxdg, Nzdg, size_MM2);
    L2_solV2 = zeros(Nyplus, Nxdg, Nzdg, size_MM2);
    L2_solW2 = zeros(Nyplus, Nxdg, Nzdg, size_MM2);

    L2_solUNN = zeros(Nyplus, Nxdg, Nzdg, size_MM2);
    L2_solVNN = zeros(Nyplus, Nxdg, Nzdg, size_MM2);
    L2_solWNN = zeros(Nyplus, Nxdg, Nzdg, size_MM2);
    Basis_eval2 = zeros(Nx_interp, Nz_interp, (p_DG2 + 1)^2);

    %Calculate Basis at interpolation points
    %Evalaute all basis at all these points
    for i = 1:(p_DG2 + 1)

        for j = 1:(p_DG2 + 1)
            nindex = (j - 1) * (p_DG2 + 1) + i;

            for i1 = 1:Nx_interp

                for j1 = 1:Nz_interp
                    Basis_eval2(i1, j1, nindex) = eval_lag2D(x_interpr(i1), z_interpr(j1), p_DG2, i, j);
                end

            end

        end

    end

    %Project to higher polynomial space to deconvolve

    %Element loop%L2 Project DNS data
    for ypi = 1:Nyplus
        yindex = yplusi(ypi);
        Uplane2 = squeeze(Usnap(:, :, ypi));
        Vplane2 = squeeze(Vsnap(:, :, ypi));
        Wplane2 = squeeze(Wsnap(:, :, ypi));
        Uplane = extend(Uplane2);
        Vplane = extend(Vplane2);
        Wplane = extend(Wplane2);

        zp = linspace(0, pi, nz + 1);
        xp = linspace(0, 2 * pi, nx + 1);

        [Z2d, X2d] = meshgrid(zp, xp);

        %HACK VARIABLE
        if (hack == 1)
            Uplane = X2d + 2 * Z2d;
        end

        %Element loop
        for i = 1:Nxdg

            for j = 1:Nzdg
                x_interp2 = x_interp + x_ref(i);
                z_interp2 = z_interp + z_ref(j);
                [Zq, Xq] = meshgrid(z_interp2, x_interp2);

                Uq = interp2(zp, xp, Uplane, Zq, Xq, 'cubic');
                Vq = interp2(zp, xp, Vplane, Zq, Xq, 'cubic');
                Wq = interp2(zp, xp, Wplane, Zq, Xq, 'cubic');

                RUq2 = zeros(size_MM2, 1);
                RVq2 = zeros(size_MM2, 1);
                RWq2 = zeros(size_MM2, 1);

                for k = 1:size_MM2
                    RUq2(k) = simpson2D(Uq .* Basis_eval2(:, :, k), dx_dg2, dz_dg2);
                    RVq2(k) = simpson2D(Vq .* Basis_eval2(:, :, k), dx_dg2, dz_dg2);
                    RWq2(k) = simpson2D(Wq .* Basis_eval2(:, :, k), dx_dg2, dz_dg2);
                end

                nodeU = MassMat2 \ RUq2;
                nodeV = MassMat2 \ RVq2;
                nodeW = MassMat2 \ RWq2;

                L2_solU2(ypi, i, j, :) = nodeU(:);
                L2_solV2(ypi, i, j, :) = nodeV(:);
                L2_solW2(ypi, i, j, :) = nodeW(:);

            end

        end

    end

    load('data/n2m_matrix2.mat');

    % %write to .h5 file for python
    % status = system('rm -f ./input_NN.h5');
    % status = system('rm -f ./output_NN.h5');
    % hdf5write('./input_NN.h5', '/input_NN', input_NN);
    % status = system('python ./fwd_run_model_simple.py');
    % output_NN = h5read('output_NN.h5','/output_NN');
    % output_NN = output_NN';

    ctr = 1;
    %Element loop%L2 Project DNS data
    for ypi = 1:Nyplus

        %Element loop
        for i = 1:Nxdg

            for j = 1:Nzdg

                im = mod(i - 1 + Nxdg - 1, Nxdg) + 1;
                ip = mod(i + 1 + Nxdg - 1, Nxdg) + 1;

                jm = mod(j - 1 + Nzdg - 1, Nzdg) + 1;
                jp = mod(j + 1 + Nzdg - 1, Nzdg) + 1;

                coeff2C = squeeze(L2_solU(ypi, i, j, :))';
                coeff2E = squeeze(L2_solU(ypi, ip, j, :))';
                coeff2W = squeeze(L2_solU(ypi, im, j, :))';
                coeff2N = squeeze(L2_solU(ypi, i, jp, :))';
                coeff2S = squeeze(L2_solU(ypi, i, jm, :))';
                coeff2SE = squeeze(L2_solU(ypi, ip, jm, :))';
                coeff2SW = squeeze(L2_solU(ypi, im, jm, :))';
                coeff2NE = squeeze(L2_solU(ypi, ip, jp, :))';
                coeff2NW = squeeze(L2_solU(ypi, im, jp, :))';

                coeff2 = [coeff2SW coeff2S coeff2SE coeff2W coeff2C coeff2E coeff2NW coeff2N coeff2NE];

                [coeff2i] = injectp(p_DG, p_DG2, coeff2C);
                coeff2i_nrm = n2m_matrix2 * (coeff2i');
                mean2 = coeff2i_nrm(1);
                emodes2 = coeff2i_nrm.^2;
                rms2 = sqrt(emodes2(2) + emodes2(5) + emodes2(6));

                coeff4r = output_NN(ctr, :);

                coeff4 = coeff4r * rms2 + coeff2i;

                L2_solUNN(ypi, i, j, :) = coeff4(:);
                if stage == 'dg'
                    L2_solU2_L(ypi, i, j, :) = coeff2i(:);
                end

                ctr = ctr + 1;
            end

        end

    end
    
    
    for ypi = 1:Nyplus
        for i = 1:Nxdg
            for j = 1:Nzdg
                coeff2 = injectp(p_DG,p_DG2,squeeze(L2_solU(2,i,j,:))'); 
                if stage == 'dg'
                    L2_solU2_L(ypi, i, j, :) = coeff2(:);
                end
            end
        end
    end

    clear U V W

    %%
    NintDG2 = Nxdg * 2;
    NintDG4 = Nxdg * 4;

    [kspec_dns, espec_dns] = spectrax(Lx, Lz, 512, 512, squeeze(squeeze(Uplane2)));
    [k2spec_dns, e2spec_dns] = spectray(Lx, Lz, 512, 512, squeeze(squeeze(Uplane2)));

    [ZfieldP2, XfieldP2, UfieldP2] = DGpolate(2 * pi, pi, NintDG2, NintDG2 / 2, 1, squeeze(L2_solU(2, :, :, :)));
    [kspec_P2, espec_P2] = spectrax(Lx, Lz, NintDG2, NintDG2 / 2, UfieldP2);
    [k2spec_P2, e2spec_P2] = spectray(Lx, Lz, NintDG2, NintDG2 / 2, UfieldP2);

    [ZfieldP4, XfieldP4, UfieldP4] = DGpolate(2 * pi, pi, NintDG4, NintDG4 / 2, 3, squeeze(L2_solU2(2, :, :, :)));
    [kspec_P4, espec_P4] = spectrax(Lx, Lz, NintDG4, NintDG4 / 2, UfieldP4);
    [k2spec_P4, e2spec_P4] = spectray(Lx, Lz, NintDG4, NintDG4 / 2, UfieldP4);

    [ZfieldP4NN, XfieldP4NN, UfieldP4NN] = DGpolate(2 * pi, pi, NintDG4, NintDG4 / 2, 3, squeeze(L2_solUNN(2, :, :, :)));
    [kspec_P4NN, espec_P4NN] = spectrax(Lx, Lz, NintDG4, NintDG4 / 2, UfieldP4NN);
    [k2spec_P4NN, e2spec_P4NN] = spectray(Lx, Lz, NintDG4, NintDG4 / 2, UfieldP4NN);

%     out_spectra_fn = "data/spectra_"+stage+z_plus+"_"+Nxdg+".mat";
%     save(out_spectra_fn, ...
%         'kspec_dns', 'espec_dns', 'kspec_P2', 'espec_P2', 'kspec_P4', 'espec_P4', 'kspec_P4NN', 'espec_P4NN', ...
%         'k2spec_dns', 'e2spec_dns', 'k2spec_P2', 'e2spec_P2', 'k2spec_P4', 'e2spec_P4', 'k2spec_P4NN', 'e2spec_P4NN');

    out_recon_fn = "data/recon_"+stage+z_plus+"_"+Nxdg+".mat";
    save(out_recon_fn);
end
