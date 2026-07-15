classdef ModelSpec
    %MODELSPEC Container for a linear-Gaussian state-space system.
    %
    %   The model is, for t = 2..T (t = 3..T where A2 is nonzero):
    %
    %       alpha_t = c_t + A1_t * alpha_{t-1} + A2_t * alpha_{t-2} + eta_t,
    %       eta_t ~ N(0, Q_t),          alpha_1 ~ N(a1, P1),
    %       y_t     = d_t + Z_t * alpha_t + eps_t,
    %       eps_t ~ N(0, diag(Rdiag_t)),
    %
    %   with missing observations coded as NaN in y and masked (never
    %   imputed).  All system matrices may be constant (one slice) or
    %   time-varying (one slice per t).
    %
    %   This class holds the numeric system at a *given* parameter value.
    %   The JointSTAR-specific mapping theta -> system lives in the static
    %   constructor ModelSpec.jointstar (state layout, model equations,
    %   COVID adjustments, variance breaks); the estimation
    %   machinery (jointstar.computeLogLik, jointstar.drawStates,
    %   jointstar.runSMC) is written against the generic system() struct.
    %   A full non-diagonal measurement covariance can be supplied via
    %   Rfull (used instead of Rdiag when nonempty).
    %
    %   See also jointstar.computeLogLik, jointstar.drawStates.

    properties
        StateNames (1, :) string = string.empty   % m state labels
        ObsNames   (1, :) string = string.empty   % p observable labels
        A1  double   % m x m (x T) first-lag transition
        A2  double = []   % m x m (x T) second-lag transition (optional)
        c   double = []   % m x 1 (or m x T) transition intercepts (optional)
        Q   double   % m x m (x T) innovation covariance, SPD
        Quniq double = []  % m x m x nGroups unique Q slices (cache-aware path
                           % only; parallels Q, see jointstar.buildEvalCache)
        a1  double   % m x 1 initial-state mean
        P1  double   % m x m initial-state covariance, SPD
        Z   double   % p x m (x T) observation loadings
        Zlag double = []  % p x m (x T) loadings on alpha_{t-1} (optional)
        d   double = []   % p x 1 (or p x T) observation intercepts (optional)
        Rdiag double = []  % p x 1 (or p x T) measurement-error variances
        Rfull double = []  % p x p (x T) full measurement covariance (optional,
                           % used instead of Rdiag when nonempty)
        Runiq double = []  % p x p x nGroups unique Rfull slices (cache-aware
                           % path only; parallels Rfull)
        T   double   % number of periods
    end

    methods
        function obj = ModelSpec(varargin)
            % Name-value construction: ModelSpec('A1', ..., 'Q', ..., ...)
            for k = 1:2:numel(varargin)
                obj.(varargin{k}) = varargin{k + 1};
            end
        end

        function sys = system(obj)
            %SYSTEM Return the plain struct consumed by computeLogLik.
            sys = struct('A1', obj.A1, 'A2', obj.A2, 'c', obj.c, ...
                'Q', obj.Q, 'Quniq', obj.Quniq, 'a1', obj.a1, 'P1', obj.P1, ...
                'Z', obj.Z, 'Zlag', obj.Zlag, 'd', obj.d, ...
                'Rdiag', obj.Rdiag, 'Rfull', obj.Rfull, 'Runiq', obj.Runiq, ...
                'T', obj.T);
        end
    end

    methods (Static)
        function [names, ix] = jointstarStates()
            %JOINTSTARSTATES State labels and index struct (m = 14).
            names = ["c", "Ustar", "pie", "zstar", "gz", "kstar", "gk", ...
                "wstar", "gw", "hppstar", "ghpp", "prstar", "gpr", "xi"];
            cn = cellstr(names);
            ix = cell2struct(num2cell(1:numel(names)), cn, 2);
        end

        function obj = jointstar(th, dat, cf, cache)
            %JOINTSTAR Build the JointSTAR system at parameter struct th.
            %
            %   obj = jointstar.ModelSpec.jointstar(th, dat)
            %   obj = jointstar.ModelSpec.jointstar(th, dat, cf)
            %   obj = jointstar.ModelSpec.jointstar(th, dat, cf, cache)
            %
            %   th  parameter struct (see jointstar.defaultPriors /
            %       jointstar.thetaStruct); dat from jointstar.loadData.
            %   cache  optional struct from jointstar.buildEvalCache.  When
            %       present, the theta-independent regime groupings for Q
            %       (COVID-kappa / break windows) and, under the
            %       horseshoe, Rfull are taken from the cache instead of
            %       rediscovered via unique() every call, and only the
            %       handful of UNIQUE slices are built (sys.Quniq /
            %       sys.Runiq, m x m x nGroups) instead of the full dense
            %       m x m x T array -- both a cycle and an allocation
            %       saving (jointstar.computeLogLik consumes Quniq/Runiq
            %       together with the same cache).  Omitted or empty =>
            %       identical to the pre-existing behaviour (sys.Q /
            %       sys.Rfull built as full T-slice arrays via unique()).
            %   cf  optional Cholesky-factor struct (Checkpoint 5+):
            %       cf.Lq (14x14) / cf.Lr (8x8; leading 7x7 submatrix
            %       used when the pi_e row is absent) unit lower
            %       triangular (Checkpoint 5+ horseshoe extension, since
            %       removed from production; not currently constructed
            %       anywhere in this codebase).  The underlying
            %       shock covariances become Lq*diag(sig.^2)*Lq' and
            %       Lr*diag(sme.^2)*Lr'.  kappa / break multipliers scale
            %       the UNDERLYING shock sds via K*Sigma*K (for the
            %       measurement side that equals scaling marginal sds;
            %       for transitions it scales the named shock, matching
            %       the Checkpoint 4 semantics of the break multipliers).
            %       Omitted => both identity (diagonal model, Checkpoint 4
            %       behaviour, bit-identical).
            %
            %   Implements the model specification (eqs 1-34 + the COVID
            %   kappa window definitions + the 1984Q1/1993Q1 volatility
            %   breaks), with the following documented choices where the
            %   specification is ambiguous (all flagged in
            %   checkpoints/QUESTIONS.md):
            %     * single IS coefficient nu (as written in eq 8);
            %     * drift persistence 0.975 / 0.95 exactly as specified;
            %     * lpr* is the participation trend pr* throughout;
            %     * COVID-block measurement equations (28-34) used for the
            %       whole sample (d_t = 0 outside 2020-23 collapses them
            %       to eqs 1-7, except wapop's AR(1), eq 34);
            %     * the IS term is dropped (not imputed) in quarters where
            %       r_{t-1} or r_{t-2} is unobservable (real cash rate
            %       starts 1993Q1);
            %     * trend/drift pairs with contemporaneous drift (e.g.
            %       z*_t = g^z_t + z*_{t-1} + eta^z) are first-order with
            %       the drift shock entering both rows, giving an exact
            %       2x2 SPD block in Q.
            if nargin < 4, cache = []; end
            useCache = ~isempty(cache);

            [~, ix] = jointstar.ModelSpec.jointstarStates();
            m = 14; T = dat.T;
            p = numel(dat.obsNames);        % 7, or 8 with the pi_e row
            hasPie = p == 8;

            % gap AR(2): theta carries (phisum, phi2) per the Rees (2019)
            % prior parameterisation; phi1 is derived
            th.phi1 = th.phisum - th.phi2;

            % r*-trend growth pair: under 'GTrendRotation' (checkpoint 13),
            % theta carries (gtrend_sum, gtrend_split) instead of
            % (gzbar, gwbar); derive the latter before ck/cz/cw are built.
            % Mirrors the phisum/phi2 -> phi1 derivation above. Absent =>
            % th.gzbar/th.gwbar are already the theta fields (unchanged
            % behaviour).
            if isfield(th, 'gtrend_sum')
                th.gzbar = (th.gtrend_sum + th.gtrend_split) / 2;
                th.gwbar = (th.gtrend_sum - th.gtrend_split) / 2;
            end

            drift975 = 0.975; drift95 = 0.95;
            iskoef = (th.nu / 2) * (4 / (1 - th.alpha));

            % ---- A1 / A2 / c (time-varying: pi^e regime, IS availability)
            A0 = zeros(m);
            A0(ix.Ustar, ix.Ustar) = 1;
            A0(ix.zstar, ix.zstar) = 1;  A0(ix.zstar, ix.gz) = drift975;
            A0(ix.gz, ix.gz) = drift975;
            A0(ix.kstar, ix.kstar) = 1;  A0(ix.kstar, ix.gk) = drift975;
            A0(ix.gk, ix.gk) = drift975;
            A0(ix.wstar, ix.wstar) = 1;  A0(ix.wstar, ix.gw) = drift975;
            A0(ix.gw, ix.gw) = drift975;
            A0(ix.hppstar, ix.hppstar) = 1; A0(ix.hppstar, ix.ghpp) = drift95;
            A0(ix.ghpp, ix.ghpp) = drift95;
            A0(ix.prstar, ix.prstar) = 1; A0(ix.prstar, ix.gpr) = drift95;
            A0(ix.gpr, ix.gpr) = drift95;
            % rate-gap AR(1) (design e4d, 'RateGapAR', DEFAULT FALSE):
            % under isfield(th,'rho_rg'), xi becomes a stationary
            % mean-zero AR(1) (A0=rho_rg) instead of a driftless random
            % walk (A0=1); P1(xi,xi) below moves to the matching
            % unconditional variance.  c(xi,:) stays 0 (mean-zero) either
            % way. Absent 'rho_rg' => bit-identical to prior behaviour.
            hasRateGapAR = isfield(th, 'rho_rg');
            if hasRateGapAR
                A0(ix.xi, ix.xi) = th.rho_rg;
            else
                A0(ix.xi, ix.xi) = 1;
            end
            A0(ix.c, ix.c) = th.phi1;

            A1 = repmat(A0, 1, 1, T);
            A2 = zeros(m, m, T);
            A2(ix.c, ix.c, :) = th.phi2;
            c = zeros(m, T);

            cz = 0.025 * th.gzbar;
            ck = 0.025 * (th.gzbar + th.gwbar) / (1 - th.alpha);
            cw = 0.025 * th.gwbar;
            c(ix.zstar, :) = cz;  c(ix.gz, :) = cz;
            c(ix.kstar, :) = ck;  c(ix.gk, :) = ck;
            c(ix.wstar, :) = cw;  c(ix.gw, :) = cw;

            % pi^e: RW before 1993Q1, AR(1) around pistar after
            A1(ix.pie, ix.pie, dat.pre93) = 1;
            A1(ix.pie, ix.pie, ~dat.pre93) = 1 - th.alphapi;
            c(ix.pie, ~dat.pre93) = th.alphapi * th.pistar;

            % IS term: c*_t += (nu/2) sum_{j=1,2} (r_{t-j} - r*_{t-j}),
            % r* = 4/(1-alpha) g^z + xi; active only where r lags observed
            on = dat.hasIS;
            A1(ix.c, ix.gz, on) = -iskoef;
            A1(ix.c, ix.xi, on) = -th.nu / 2;
            A2(ix.c, ix.gz, on) = -iskoef;
            A2(ix.c, ix.xi, on) = -th.nu / 2;
            c(ix.c, on) = (th.nu / 2) * (dat.r1(on) + dat.r2(on));

            % ---- Q: underlying shocks eta with cov Lq*D*Lq', scaled per
            % t by K (breaks + COVID kappa), mapped to state innovations
            % by M (contemporaneous drift => trend row shock includes the
            % drift shock).  Diagonal Lq reproduces the exact 2x2
            % trend/drift blocks of the specification's timing.
            s2 = @(x) x.^2;
            if nargin < 3 || isempty(cf)
                Lq = eye(m); Lr = eye(p);
            else
                Lq = cf.Lq;
                % pie_obs is last in the Cholesky ordering, so when the
                % row is absent the leading submatrix of Lr is exactly
                % the factor of the remaining errors' covariance
                Lr = cf.Lr(1:p, 1:p);
            end
            dq = zeros(m, 1);
            dq(ix.c) = th.sig_c; dq(ix.Ustar) = th.sig_Ustar;
            dq(ix.pie) = th.sig_pie;
            dq(ix.zstar) = th.sig_z; dq(ix.gz) = th.sig_gz;
            dq(ix.kstar) = th.sig_k; dq(ix.gk) = th.sig_gk;
            dq(ix.wstar) = th.sig_w; dq(ix.gw) = th.sig_gw;
            dq(ix.hppstar) = th.sig_hpp; dq(ix.ghpp) = th.sig_ghpp;
            dq(ix.prstar) = th.sig_pr; dq(ix.gpr) = th.sig_gpr;
            dq(ix.xi) = th.sig_xi;
            Sig0 = Lq * diag(s2(dq)) * Lq';
            Sig0 = (Sig0 + Sig0') / 2;

            M = eye(m);
            M(ix.zstar, ix.gz) = 1; M(ix.kstar, ix.gk) = 1;
            M(ix.wstar, ix.gw) = 1; M(ix.hppstar, ix.ghpp) = 1;
            M(ix.prstar, ix.gpr) = 1;

            kq = ones(m, T);
            kq(ix.c, dat.pre84) = th.m84_c;
            kq(ix.zstar, dat.pre84) = th.m84_z;
            kq(ix.Ustar, dat.pre93) = th.m93_U;
            kq(ix.c, dat.win.w2020_21) = th.kapc_2021;

            if useCache
                % regime grouping precomputed once (buildEvalCache); build
                % only the ~nGQ unique slices, skip the full T-slice
                % dense array and the per-eval unique() rediscovery.
                nGQ = cache.nGQ;
                Q = [];
                Quniq = zeros(m, m, nGQ);
                for uu = 1:nGQ
                    kk = kq(:, cache.QgroupRep(uu));
                    Qu = M * (Sig0 .* (kk * kk')) * M';
                    Quniq(:, :, uu) = (Qu + Qu') / 2;
                end
            else
                Q = zeros(m, m, T);
                [kuq, ~, iuq] = unique(kq', 'rows');
                for uu = 1:size(kuq, 1)
                    kk = kuq(uu, :)';
                    Qu = M * (Sig0 .* (kk * kk')) * M';
                    Q(:, :, iuq == uu) = repmat((Qu + Qu') / 2, 1, 1, nnz(iuq == uu));
                end
                Quniq = [];
            end

            % ---- measurement loadings (constant) ------------------------
            Z = zeros(p, m); ZL = zeros(p, m);
            a = th.alpha;
            % 1: y = tau* + c* + eps;  tau* = z* + a k* + (1-a) h*
            Z(1, ix.c) = 1; Z(1, ix.zstar) = 1; Z(1, ix.kstar) = a;
            Z(1, [ix.wstar, ix.prstar, ix.hppstar]) = 1 - a;
            Z(1, ix.Ustar) = -(1 - a);
            % 2: Phillips
            Z(2, ix.pie) = th.gamma1; Z(2, ix.Ustar) = -th.gamma2;
            % 3: wapop (eq 34)
            Z(3, ix.wstar) = 1; ZL(3, ix.wstar) = -th.rhow;
            % 4-7: U, lpr, hpp, k (eqs 30-33)
            Z(4, ix.Ustar) = 1; Z(4, ix.c) = th.xi1;
            ZL(4, ix.c) = th.xi2; ZL(4, ix.Ustar) = -th.rhoU;
            Z(5, ix.prstar) = 1; Z(5, ix.c) = th.theta1;
            ZL(5, ix.c) = th.theta2; ZL(5, ix.prstar) = -th.rhopr;
            Z(6, ix.hppstar) = 1; Z(6, ix.c) = th.lam1;
            ZL(6, ix.c) = th.lam2; ZL(6, ix.hppstar) = -th.rhohpp;
            Z(7, ix.kstar) = 1; Z(7, ix.c) = th.chi1;
            ZL(7, ix.c) = th.chi2; ZL(7, ix.kstar) = -th.rhok;
            % 8 (Checkpoint 7): pi_e observed = trend inflation + noise
            if hasPie
                Z(8, ix.pie) = 1;
            end

            % ---- measurement intercepts (data + COVID shifters) ---------
            g = dat.ing; D = dat.d; DL = g.Dlag;
            d = zeros(p, T);
            d(1, :) = th.phiy * D;
            d(2, :) = (1 - th.gamma1) * g.PIlag + th.gamma2 * (g.Ulev + th.phiu * D);
            d(3, :) = th.rhow * g.Wlag;
            d(4, :) = -th.phiu * D + th.rhoU * (g.Ulag + th.phiu * DL);
            d(5, :) = th.phipr * D + th.rhopr * (g.LPRlag - th.phipr * DL);
            d(6, :) = th.phihpp * D + th.rhohpp * (g.HPlag - th.phihpp * DL);
            d(7, :) = th.phik * D + th.rhok * (g.Klag - th.phik * DL);

            % ---- measurement variances: breaks + COVID kappas ----------
            % kr holds per-variable sd multipliers; correlations from Lr
            % are preserved under the K*R0*K scaling
            w = dat.win;
            kr = ones(p, T);
            kr(1, dat.pre84) = th.m84_y;
            kr(2, dat.pre93) = th.m93_pi;
            kr(1, w.w2020) = th.kapy_20;    kr(1, w.w2021) = th.kapy_21;
            kr(2, w.w2020_23) = th.kappi_2023;
            kr(3, w.w2020_21) = th.kappop_2021;
            kr(4, w.w2020) = th.kapu_20;    kr(4, w.w2021_22) = th.kapu_2122;
            kr(5, w.w2020) = th.kappr_20;   kr(5, w.w2021_22) = th.kappr_2122;
            kr(6, w.w2020_22) = th.kaphpp_2022;
            kr(7, w.w2020) = th.kapk_20;    kr(7, w.w2021) = th.kapk_21;
            % pi_e row (if present): no COVID kappa, no break multiplier

            sme = [th.sme_y; th.sme_pi; th.sme_w; th.sme_U; ...
                th.sme_pr; th.sme_hpp; th.sme_k];
            if hasPie
                sme = [sme; th.sme_pieobs];
            end
            diagR = nargin < 3 || isempty(cf);
            Runiq = [];
            if diagR
                Rd = s2(sme) .* s2(kr);     % fast diagonal path
                Rfull = [];
            else
                R0 = Lr * diag(s2(sme)) * Lr';
                R0 = (R0 + R0') / 2;
                Rd = [];
                if useCache
                    nGR = cache.nGR;
                    Rfull = [];
                    Runiq = zeros(p, p, nGR);
                    for uu = 1:nGR
                        kk = kr(:, cache.RgroupRep(uu));
                        Ru = R0 .* (kk * kk');
                        Runiq(:, :, uu) = (Ru + Ru') / 2;
                    end
                else
                    Rfull = zeros(p, p, T);
                    [kur, ~, iur] = unique(kr', 'rows');
                    for uu = 1:size(kur, 1)
                        kk = kur(uu, :)';
                        Ru = R0 .* (kk * kk');
                        Rfull(:, :, iur == uu) = ...
                            repmat((Ru + Ru') / 2, 1, 1, nnz(iur == uu));
                    end
                end
            end

            % ---- initial state ------------------------------------------
            an = dat.anchor;
            a1 = zeros(m, 1);
            a1(ix.Ustar) = an.U0; a1(ix.pie) = an.PIE0;
            a1(ix.kstar) = an.K0; a1(ix.wstar) = an.W0;
            a1(ix.hppstar) = an.HP0; a1(ix.prstar) = an.LPR0;
            a1(ix.zstar) = an.Y0 - a * an.K0 ...
                - (1 - a) * (an.W0 + an.LPR0 + an.HP0 - an.U0);
            a1(ix.gz) = 0.3; a1(ix.gk) = 1.0; a1(ix.gw) = 0.5;
            a1(ix.ghpp) = -0.1; a1(ix.gpr) = 0.1;
            P1 = diag([25, 25, 25, 100, 0.25, 100, 0.25, 100, 0.25, ...
                100, 0.25, 100, 0.25, 4]);
            if hasRateGapAR
                % stationary AR(1) unconditional variance; xi has no
                % cross-loadings and an independent diagonal innovation
                % (Sig0/M leave xi's row/column untouched), so this is
                % exact -- no initial cross-covariance needed. Mirrors
                % ModelSpec.toyAR1's own convention.
                P1(ix.xi, ix.xi) = th.sig_xi^2 / (1 - th.rho_rg^2);
            end

            obj = jointstar.ModelSpec('A1', A1, 'A2', A2, 'c', c, ...
                'Q', Q, 'Quniq', Quniq, 'a1', a1, 'P1', P1, 'Z', Z, 'Zlag', ZL, ...
                'd', d, 'Rdiag', Rd, 'Rfull', Rfull, 'Runiq', Runiq, 'T', T, ...
                'StateNames', jointstar.ModelSpec.jointstarStates(), ...
                'ObsNames', string(dat.obsNames));
        end

        function obj = toyAR1(rho, sigEta, sigEps, T, a1, P1)
            %TOYAR1 Scalar AR(1) + noise toy used by Checkpoint 1 tests.
            %   alpha_t = rho*alpha_{t-1} + eta_t,  y_t = alpha_t + eps_t.
            if nargin < 5, a1 = 0; end
            if nargin < 6, P1 = sigEta^2 / (1 - rho^2); end
            obj = jointstar.ModelSpec('A1', rho, 'Q', sigEta^2, ...
                'a1', a1, 'P1', P1, 'Z', 1, 'Rdiag', sigEps^2, 'T', T);
        end
    end
end
