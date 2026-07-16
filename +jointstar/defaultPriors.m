function P = defaultPriors(varargin)
%DEFAULTPRIORS Prior specification for the JointSTAR parameter vector.
%
%   P = jointstar.defaultPriors()
%   P = jointstar.defaultPriors('HierKappa', true)
%
%   With 'HierKappa' (Checkpoint 6), the 12 COVID variance scale factors
%   get the hierarchical prior of the project brief instead of
%   independent truncated Gammas:
%
%       kappa_v ~ Gamma(a_g, b_g) * 1[kappa >= 1]
%
%   with (a_g, b_g) SHARED across variables within the same time-window
%   group g and themselves estimated (log-mean and log-shape carried in
%   theta with Normal priors; the truncation normaliser, which now
%   depends on the hyperparameters, is included in the density).  This
%   shrinks the kappas toward common within-window values.  Groups:
%   2020 = {y, u, pr, k}; 2021 = {y, k}; 2021-22 = {u, pr};
%   2020-21 = {c, pop}; and the two single-member windows 2020-22 (hpp)
%   and 2020-23 (pi), where the hierarchy reduces to a loose prior.
%
%   Dynamics priors follow the Rees (2019) model-specification priors
%   (2026-07-10 ruling), including two parameterisation conventions
%   confirmed with the model owner:
%     * the Phillips "unemployment gap" coefficient enters as
%       gamma2 = -|gamma2| with |gamma2| ~ Beta(mean .2, sd .1) -- the
%       reported Beta has a negative mean, which only makes sense as a
%       sign-flipped parameterisation of a standard [0,1]-support Beta;
%     * the gap AR(2) is parameterised as (phisum, phi2) with
%       phisum = phi1 + phi2 ~ Beta(mean .5, sd .29) and
%       phi2 ~ N(-.5, 1); phi1 is derived -- the source table gives a
%       Normal prior on phi2 and a memo Beta on the sum, implying the
%       (sum, second-lag) parameterisation used here;
%     * the "output gap (t)+(t-1)" Normal priors are placed on the SUM of
%       the two loadings; here each loading gets N(mean/2, sd/sqrt(2)) so
%       the implied prior on the sum matches the source exactly.
%   The source priors are a "selected subset": everything they omit --
%   shock sds, break multipliers, pistar/alphapi, COVID phis and kappas,
%   and the capital/wapop equation coefficients -- keeps the documented
%   weakly-informative defaults below, flagged in checkpoints/QUESTIONS.md.
%
%   Types:
%     norm    N(p1, p2)                      (p2 = sd)
%     tnorm   N(p1, p2) truncated to [lo, hi]
%     unif    U(lo, hi)
%     beta    Beta(p1, p2) on (0, 1)         (shape parameters)
%     negbeta x in (-1, 0) with |x| ~ Beta(p1, p2)
%     igsd    sd sigma with sigma^2 ~ InvGamma(p1, p2); parameterised
%             here by shape p1 and target sd p2, b = (p1-1)*p2^2, so
%             E[sigma^2] = p2^2.  Keeps variances softly away from zero
%             (the HLW pile-up point) without forbidding small values.
%     logn    logNormal(p1, p2) (multipliers)
%     tgamma  Gamma(shape p1, scale p2) truncated to [lo, Inf) — COVID
%             kappas, lo = 1 (rejection sampling; truncation normaliser
%             omitted from the log-density: constant in theta)
%
%   A joint AR(2) stationarity constraint on the implied (phi1, phi2) is
%   enforced in priorLogPdf / priorSample.
%
%   See also jointstar.priorLogPdf, jointstar.priorSample.

k = 0; prm = struct('name', {}, 'type', {}, 'p1', {}, 'p2', {}, ...
    'lo', {}, 'hi', {}, 'init', {});
    function add(name, type, p1, p2, lo, hi, init)
        k = k + 1;
        prm(k) = struct('name', name, 'type', type, 'p1', p1, 'p2', p2, ...
            'lo', lo, 'hi', hi, 'init', init);
    end

% 'GTrendRotation' (Checkpoint 13, DEFAULT FALSE): rotate the r*-trend
% growth pair (gzbar, gwbar) -- pooled corr -0.83, worst cross-seed R-hats
% under the raw kernel -- onto (gtrend_sum, gtrend_split) = (gzbar+gwbar,
% gzbar-gwbar), mirroring the existing (phisum, phi2) gap-AR rotation.
% ModelSpec.jointstar's ck = 0.025*(gzbar+gwbar)/(1-alpha) means the model
% mainly identifies the SUM, so this puts the RW proposal on the
% identified/unidentified axes directly.  Since gzbar ~ N(.30,.15) and
% gwbar ~ N(.40,.15) have EQUAL sds, the implied (sum, split) prior is
% exactly (not approximately) independent Normal: for independent
% Normals, Cov(sum,split) = Var(gzbar) - Var(gwbar) = 0 when the sds
% match, so no prior-equivalence approximation is involved here (see
% notes for the general s1~=s2 case).  DEFAULT FALSE emits the original
% gzbar/gwbar rows unchanged.
ipr = inputParser;
ipr.addParameter('GTrendRotation', false);
ipr.KeepUnmatched = true;
ipr.parse(varargin{:});
useGTrendRotation = ipr.Results.GTrendRotation;

% 'RateGapAR' (design e4d, DEFAULT FALSE): switch the xi state (the
% non-trend-growth component of r*, r*_t = 4/(1-alpha)*gz_t + xi_t) from
% a driftless random walk (A0(xi,xi)=1, P1(xi,xi)=4 fixed) to a
% stationary mean-zero AR(1), A0(xi,xi)=rho_rg, P1(xi,xi)=
% sig_xi^2/(1-rho_rg^2).  This removes the permanent-shock (unit-root)
% component the RW currently contributes to the IS equation's rate-gap
% forcing term (nu/2)*sum_j(r_{t-j}-r*_{t-j}), which today inherits an
% accumulating xi piece -- the owner's "make the [RW piece of] the real
% rate gap an AR process" instruction.  sig_xi, the IS-equation loadings
% on xi, and the r1/r2 exogenous forcing are all UNCHANGED (see
% ModelSpec.jointstar).  DEFAULT FALSE => isfield(th,'rho_rg') is false
% => A0(xi,xi)=1, P1(xi,xi)=4 exactly as before, bit-identical.
ipg = inputParser;
ipg.addParameter('RateGapAR', false);
ipg.KeepUnmatched = true;
ipg.parse(varargin{:});
useRateGapAR = ipg.Results.RateGapAR;

% 'DropPhiY' (DEFAULT FALSE): a nested-restriction hypothesis test of the
% COVID level shifter on GDP (phiy).  When DropPhiY=false (default), phiy
% is sign-restricted < 0 via truncated normal N(0, 0.10) on (-Inf, 0).
% When DropPhiY=true, phiy is fixed at 0 (a 'fixed' prior type, excluded
% from mutateIdx), dropping the GDP COVID intercept term entirely from
% d_t; this tests whether the stringency loading is redundant GIVEN kappa
% (the d_t/kappa competition on the 2020 GDP drop, CHECKPOINT_07).  This
% is a NESTED RESTRICTION (H0: phiy=0 within H1: phiy<0), not a sign-flip
% reversal.  Composes cleanly with all other flags (orthogonal); phiu's
% sign restriction (< 0) is unchanged. Parameter count 79 -> 78 (free
% params 78 -> 77) when flag is true.  DEFAULT FALSE reproduces the prior
% phiy row byte-for-byte.
ipd = inputParser;
ipd.addParameter('DropPhiY', false);
ipd.KeepUnmatched = true;
ipd.parse(varargin{:});
useDropPhiY = ipd.Results.DropPhiY;

% ---- dynamics (Rees 2019 priors where reported) ------------------------
% Beta(a,b) from (mean m, sd s): c = m(1-m)/s^2 - 1, a = m*c, b = (1-m)*c
add('gamma1', 'beta', 3.0, 2.0, 0, 1, 0.6);    % pi^e weight: B(m=.6, s=.2)
add('gamma2', 'negbeta', 3.0, 12.0, -1, 0, -0.2); % -|g2|, |g2| ~ B(.2, .1)
add('xi1',    'norm', -0.25, 0.7071, -Inf, Inf, -0.25); % sum ~ N(-.5, 1)
add('xi2',    'norm', -0.25, 0.7071, -Inf, Inf, -0.25);
add('rhoU',   'beta', 0.889, 0.889, 0, 1, 0.50);        % B(m=.5, s=.3)
add('theta1', 'norm',  0.25, 0.7071, -Inf, Inf,  0.25); % sum ~ N(.5, 1)
add('theta2', 'norm',  0.25, 0.7071, -Inf, Inf,  0.25);
add('rhopr',  'beta', 0.889, 0.889, 0, 1, 0.50);
add('lam1',   'norm',  0.25, 0.7071, -Inf, Inf,  0.25); % sum ~ N(.5, 1)
add('lam2',   'norm',  0.25, 0.7071, -Inf, Inf,  0.25);
add('rhohpp', 'beta', 0.889, 0.889, 0, 1, 0.50);
% capital / wapop equations absent from the source priors: neutral analogues
add('chi1',   'norm',  0.00, 0.7071, -Inf, Inf,  0.00);
add('chi2',   'norm',  0.00, 0.7071, -Inf, Inf,  0.00);
add('rhok',   'beta', 0.889, 0.889, 0, 1, 0.50);
add('rhow',   'beta', 0.889, 0.889, 0, 1, 0.50);
if useRateGapAR
    % rate-gap AR(1) persistence (design e4d).  Beta(9.99,1.76) =>
    % mean 0.85, sd 0.10: centred below the slower financial-cycle
    % literature range (~0.90-0.97, e.g. DGGT19's convenience-yield
    % factor) so ~132 quarters of real-rate data (1993Q1+) can pull it
    % away from a de facto unit root, while (0,1) Beta support enforces
    % stationarity automatically (no joint constraint code needed, unlike
    % phi1/phi2).
    add('rho_rg', 'beta', 9.99, 1.76, 0, 1, 0.85);
end
% gap AR(2) in (sum, second-lag) parameterisation; phi1 = phisum - phi2
add('phisum', 'beta', 0.986, 0.986, 0, 1, 0.75);        % B(m=.5, s=.29)
add('phi2',   'norm', -0.50, 1.00, -Inf, Inf, -0.40);
add('nu',     'norm', -0.10, 0.15, -Inf, Inf, -0.10); % IS slope
add('alpha',  'tnorm', 0.35, 0.05, 0.15, 0.55, 0.35); % capital share
if ~useGTrendRotation
    add('gzbar',  'norm',  0.30, 0.15, -Inf, Inf,  0.30); % mean MFP drift, %/q
    add('gwbar',  'norm',  0.40, 0.15, -Inf, Inf,  0.40); % mean wapop drift, %/q
else
    % implied-exact rotation of N(.30,.15)/N(.40,.15) (equal sds => the
    % (sum,split) pair is exactly independent Normal, see note above):
    %   mean_sum = .30+.40 = .70,  var_sum = .15^2+.15^2 = .045
    %   mean_split = .30-.40 = -.10, var_split = .15^2+.15^2 = .045
    add('gtrend_sum',   'norm',  0.70, sqrt(2) * 0.15, -Inf, Inf,  0.70);
    add('gtrend_split', 'norm', -0.10, sqrt(2) * 0.15, -Inf, Inf, -0.10);
end
add('pistar', 'norm',  2.50, 0.50, -Inf, Inf,  2.50); % inflation target
add('alphapi', 'unif', 0, 0, 0.01, 0.99, 0.10);       % pi^e pull to target
% COVID level shifters.  phi_y, phi_u sign-restricted to < 0 per the
% model specification and the owner's CP7 ruling (2026-07-10):
% unrestricted, they flip sign when the pi_e anchor re-attributes the
% Phillips block (the d_t/kappa interplay documented in CHECKPOINT_07.md).
if ~useDropPhiY
    add('phiy',   'tnorm', 0, 0.10, -Inf, 0, -0.05);   % unchanged default
else
    add('phiy',   'fixed', 0, 0, 0, 0, 0);             % fixed at 0 (nested restriction)
end
add('phiu',   'tnorm', 0, 0.05, -Inf, 0, -0.02);
add('phipr',  'norm', 0, 0.05, -Inf, Inf, -0.02);
add('phihpp', 'norm', 0, 0.10, -Inf, Inf, -0.05);
add('phik',   'norm', 0, 0.05, -Inf, Inf,  0.00);

% ---- transition shock sds ----------------------------------------------
add('sig_c',    'igsd', 3, 0.80, 0, Inf, 0.80);
add('sig_Ustar', 'igsd', 3, 0.15, 0, Inf, 0.15);
add('sig_pie',  'igsd', 3, 0.30, 0, Inf, 0.30);
add('sig_z',    'igsd', 3, 0.50, 0, Inf, 0.50);
add('sig_gz',   'igsd', 3, 0.05, 0, Inf, 0.05);   % pile-up prone
add('sig_k',    'igsd', 3, 0.30, 0, Inf, 0.30);
add('sig_gk',   'igsd', 3, 0.05, 0, Inf, 0.05);
add('sig_w',    'igsd', 3, 0.20, 0, Inf, 0.20);
add('sig_gw',   'igsd', 3, 0.05, 0, Inf, 0.05);
add('sig_hpp',  'igsd', 3, 0.40, 0, Inf, 0.40);
add('sig_ghpp', 'igsd', 3, 0.05, 0, Inf, 0.05);
add('sig_pr',   'igsd', 3, 0.30, 0, Inf, 0.30);
add('sig_gpr',  'igsd', 3, 0.05, 0, Inf, 0.05);
add('sig_xi',   'igsd', 3, 0.15, 0, Inf, 0.15);   % pile-up prone

% ---- measurement error sds ---------------------------------------------
add('sme_y',   'igsd', 3, 0.60, 0, Inf, 0.60);
add('sme_pi',  'igsd', 3, 1.00, 0, Inf, 1.00);
add('sme_w',   'igsd', 3, 0.20, 0, Inf, 0.20);
add('sme_U',   'igsd', 3, 0.25, 0, Inf, 0.25);
add('sme_pr',  'igsd', 3, 0.30, 0, Inf, 0.30);
add('sme_hpp', 'igsd', 3, 0.50, 0, Inf, 0.50);
add('sme_k',   'igsd', 3, 0.20, 0, Inf, 0.20);
% Checkpoint 7: pi_e as direct measurement of trend inflation.  FIXED at
% 30bp per the owner's CP7 ruling: estimating it let the anchor claim
% ~15bp accuracy, over-trusting a constructed series and destabilising
% the COVID level shifters (CHECKPOINT_07.md).
add('sme_pieobs', 'fixed', 0, 0, 0.30, 0.30, 0.30);

% ---- pre-break volatility multipliers ----------------------------------
% 1984Q1: GDP meas., productivity, gap shocks; 1993Q1: inflation meas., NAIRU
add('m84_y',  'logn', log(1.5), 0.5, 0, Inf, 1.5);
add('m84_z',  'logn', log(1.5), 0.5, 0, Inf, 1.5);
add('m84_c',  'logn', log(1.5), 0.5, 0, Inf, 1.5);
add('m93_pi', 'logn', log(1.5), 0.5, 0, Inf, 1.5);
add('m93_U',  'logn', log(1.5), 0.5, 0, Inf, 1.5);

% ---- COVID variance scale factors (Table 2: 12 of them, confirmed by
% the model owner), all truncated >= 1 -----------------------------------
ipk = inputParser;
ipk.addParameter('HierKappa', false);
ipk.KeepUnmatched = true;
ipk.parse(varargin{:});

% 'FixSingletonKappa' (DEFAULT FALSE, only meaningful when HierKappa is
% true): the two single-member COVID-kappa window groups, g5 = w2022tot
% = {kaphpp_2022} and g6 = w2023tot = {kappi_2023}, have a hierarchical
% (a_g, b_g) pair whose hypers carry ZERO likelihood curvature -- they
% enter only priorLogPdf, never ModelSpec / the state-space likelihood
% (session-verified, CHECKPOINT_14) -- so the hierarchy there is just a
% 2-dim-per-group, likelihood-flat construction (contraction ratios
% 0.91 / 1.34: WIDER than the prior). This flag removes those 4 hyper
% dimensions (kapHyp_lm/la_w2022tot, kapHyp_lm/la_w2023tot) from theta
% and gives kaphpp_2022 / kappi_2023 a fixed truncated-Gamma prior
% instead, calibrated at the CONDITIONAL prior implied by the hypers'
% own prior means: a = exp(log 2.0) = 2.0, m = exp(log 2.5) = 2.5,
% scale s = m/a = 1.25 (same [1, Inf) truncation, same paramTransform
% kind-2 support -- both hkap and tgamma map lo=1,hi=Inf the same way).
% The multi-member groups g1..g4 are untouched: they stay 'hkap' with
% their hyper pairs still free.  DEFAULT FALSE (or HierKappa=false)
% reproduces the current add() sequence byte-for-byte: the singleton-
% detection remap below reduces to the identity (all 6 groups kept, in
% original order), so P.kap / P.mutateIdx / P.names are unchanged.
ipf = inputParser;
ipf.addParameter('FixSingletonKappa', false);
ipf.KeepUnmatched = true;
ipf.parse(varargin{:});
useFixSingleton = ipk.Results.HierKappa && ipf.Results.FixSingletonKappa;

% 'PoolAcuteOnly' (DEFAULT FALSE, only meaningful when HierKappa is true;
% INDEPENDENT of FixSingletonKappa -- own inputParser, own name): keeps
% the kappa hierarchy ONLY for g1 = w2020 (grpNames{1}, the 4-member
% acute window {kapy_20, kapu_20, kappr_20, kapk_20}) and collapses ALL
% OTHER groups -- g2(w2021), g3(w2122), g4(w2021tot), g5(w2022tot),
% g6(w2023tot) -- to the same fixed a-priori truncated-Gamma(2.0, 1.25)
% calibration FixSingletonKappa already uses for the two singletons
% (g5, g6): the conditional prior implied by the hypers' own prior
% means (a = exp(log 2.0) = 2.0, m = exp(log 2.5) = 2.5, s = m/a =
% 1.25). This removes 10 hyperparameters (5 groups x {lm, la}) from
% theta (79 -> 69 under HierKappa), re-typing 8 kappas (kapy_21,
% kapk_21, kapu_2122, kappr_2122, kapc_2021, kappop_2021, kaphpp_2022,
% kappi_2023) from 'hkap' to fixed 'tgamma' -- still FREE parameters,
% still in mutateIdx.  Only g1's hierarchy (and its hyper pair) remains.
%
% Precedence when both flags are set: PoolAcuteOnly DOMINATES
% FixSingletonKappa (it is a strict superset -- it also fixes g2, g3,
% g4, which FixSingletonKappa leaves hierarchical).  Implemented as
% if/elseif below so PoolAcuteOnly=true alone gives keep-only-g1
% regardless of FixSingletonKappa, and FixSingletonKappa=true alone
% (with PoolAcuteOnly left false) gives keep-4-groups exactly as before
% -- clean independent attribution.
%
% Honest caveat (conditional vs marginal, same as FixSingletonKappa):
% fixing at the conditional-at-hyper-means makes each pooled kappa's
% prior slightly narrower than the true hierarchical marginal (which
% integrates over hyper uncertainty).  The justification is that a
% 2-member (or 1-member) group barely updates a 2-parameter hyperprior
% from the data, so the hierarchical marginal sits very close to this
% conditional anyway -- the hierarchy buys almost no shrinkage there
% while costing 2 unidentified hyper dims per group (the mechanism
% CHECKPOINT_14 documented for the singletons, one notch weaker for the
% 2-member groups).
%
% DEFAULT FALSE (or HierKappa=false) reproduces today's add() sequence
% byte-for-byte: keepGrp falls through unchanged (all-true, or whatever
% FixSingletonKappa alone sets).
ipp = inputParser;
ipp.addParameter('PoolAcuteOnly', false);
ipp.KeepUnmatched = true;
ipp.parse(varargin{:});
usePoolAcute = ipk.Results.HierKappa && ipp.Results.PoolAcuteOnly;

% 'CovidDecay' (DEFAULT FALSE, only meaningful when HierKappa is true;
% MUTUALLY EXCLUSIVE with PoolAcuteOnly/FixSingletonKappa -- own
% inputParser, own name -- they are alternative kappa reparameterizations
% of the same groups): replaces the 8 COVID kappas belonging to the
% acute two-step hierarchical groups g1(w2020), g2(w2021), g3(w2122) --
% kapy_20/kapy_21 (row1 GDP), kapu_20/kapu_2122 (row4 U), kappr_20/
% kappr_2122 (row5 participation), kapk_20/kapk_21 (row7 capital) -- with
% a parametric geometric-decay SD multiplier (Lenza-Primiceri style):
% kr(row,t) = 1 + s0_row * rho_c^(t-t0), t0 = 2020Q2, over each row's
% original union window (dat.win.w2020_21 for rows 1/7, dat.win.w2020_22
% for rows 4/5; both already start at 2020Q2), kr = 1 outside (see
% jointstar.ModelSpec.jointstar). NEW PARAMS (5): s0_y, s0_u, s0_pr,
% s0_k ('logn', peak-excess SD amplitude, one per row) and a SINGLE
% shared decay rate rho_c ('beta'). The 4 kept groups g4(w2021tot =
% {kapc_2021, kappop_2021}), g5(w2022tot = {kaphpp_2022}), g6(w2023tot =
% {kappi_2023}) are UNCHANGED -- no multiplicity to subsume there, and
% leaving inflation/cycle free avoids touching the Phillips block/
% latent-cycle economics. Parameter count 79 -> 70 under HierKappa
% (79 - 8 kappas - 6 hypers + 5 decay). DEFAULT FALSE (or
% HierKappa=false) reproduces today's add() sequence and P.kap
% byte-for-byte -- the whole CovidDecay code path is a no-op.
ipc = inputParser;
ipc.addParameter('CovidDecay', false);
ipc.KeepUnmatched = true;
ipc.parse(varargin{:});
useCovidDecay = ipk.Results.HierKappa && ipc.Results.CovidDecay;
if useCovidDecay && (usePoolAcute || useFixSingleton)
    error('jointstar:covidDecayConflict', ...
        ['CovidDecay is mutually exclusive with PoolAcuteOnly and ' ...
         'FixSingletonKappa -- all three are alternative kappa ' ...
         'reparameterizations of the same window groups.']);
end
if useCovidDecay
    % peak-excess SD amplitude per subsumed row, logNormal(log 1.5, 0.6):
    % median s0=1.5 => median PEAK kappa = 1+1.5 = 2.5, matching the
    % incumbent HierKappa kappa prior's median (group mean m_g ~
    % exp(N(log 2.5, 0.5)), median 2.5) -- an a-priori calibration, not
    % fit to any posterior/contraction diagnostic. 90% interval s0 in
    % [0.56, 4.0] => peak kappa in [1.56, 5.0].
    add('s0_y',  'logn', log(1.5), 0.6, 0, Inf, 1.5);
    add('s0_u',  'logn', log(1.5), 0.6, 0, Inf, 1.5);
    add('s0_pr', 'logn', log(1.5), 0.6, 0, Inf, 1.5);
    add('s0_k',  'logn', log(1.5), 0.6, 0, Inf, 1.5);
    % shared geometric decay rate, Beta(mean 0.6, sd 0.15) -> Beta(a,b)
    % via c = m(1-m)/s^2 - 1: a = 5.80, b = 3.87. Half-life at the prior
    % mean ln(0.5)/ln(0.6) ~= 1.36 quarters (~4 months); 90% mass rho in
    % ~[0.34, 0.82] => half-life ~0.9-3.5 quarters, consistent with COVID
    % measurement distortions fading within about a year. Centred well
    % below 1 so the shortest subsumed window's (GDP/capital, ending
    % 2021Q4, expo=6) forced-to-1 cap sits near-continuous at the prior
    % mean (rho^6 ~= 0.047 => kappa ~= 1.07 just before the cap).
    add('rho_c', 'beta', 5.80, 3.87, 0, 1, 0.6);
end

kap = {'kapc_2021', 'kapy_20', 'kapy_21', 'kapu_20', 'kapu_2122', ...
    'kappr_20', 'kappr_2122', 'kaphpp_2022', 'kappop_2021', ...
    'kappi_2023', 'kapk_20', 'kapk_21'};
% time-window group of each kappa (order matches kap):
%          c    y20  y21  u20  u2122 pr20 pr2122 hpp  pop  pi   k20  k21
kapGrp = [ 4,   1,   2,   1,   3,    1,   3,     5,   4,   6,   1,   2 ]';
grpNames = {'w2020', 'w2021', 'w2122', 'w2021tot', 'w2022tot', 'w2023tot'};

if ~ipk.Results.HierKappa
    for j = 1:numel(kap)
        add(kap{j}, 'tgamma', 2, 2, 1, Inf, 2.0);
    end
    P = struct('params', prm, 'names', {{prm.name}}, 'd', numel(prm));
    P.idx = cell2struct(num2cell(1:P.d), P.names, 2);
    P.mutateIdx = ~strcmp({prm.type}, 'fixed');
    return
end

G = numel(grpNames);
isSingletonGrp = accumarray(kapGrp, 1) == 1;
keepGrp = true(G, 1);
if usePoolAcute
    % keep ONLY g1 = w2020 (the 4-member acute window); PoolAcuteOnly
    % dominates FixSingletonKappa when both are set (strict superset)
    keepGrp = false(G, 1);
    keepGrp(1) = true;
elseif useFixSingleton
    keepGrp = ~isSingletonGrp;
elseif useCovidDecay
    % drop g1(w2020), g2(w2021), g3(w2122) -- the acute two-step groups
    % subsumed by the geometric decay parameters above; keep g4(w2021tot),
    % g5(w2022tot), g6(w2023tot) hierarchical exactly as before (mutually
    % exclusive with usePoolAcute/useFixSingleton, guarded above)
    keepGrp = false(G, 1);
    keepGrp([4, 5, 6]) = true;
end
remap = zeros(G, 1);
remap(keepGrp) = 1:nnz(keepGrp);

% fixed truncated-Gamma calibration for the singleton kappas (a-priori
% only -- the conditional prior at the hyperpriors' own means, see the
% flag doc above; NOT informed by any posterior/contraction diagnostic)
aFix = exp(log(2.0));   % = 2.0
mFix = exp(log(2.5));   % = 2.5
sFix = mFix / aFix;     % = 1.25

kapCols = zeros(0, 1);
kapGrpKept = zeros(0, 1);
for j = 1:numel(kap)
    g = kapGrp(j);
    if ~keepGrp(g)
        if useCovidDecay
            % subsumed by the CovidDecay geometric-decay parameters
            % (s0_*/rho_c added above) -- not part of theta at all,
            % unlike the FixSingletonKappa/PoolAcuteOnly pooling below
            continue
        end
        % pooled/fixed kappa (either a FixSingletonKappa singleton, or,
        % under PoolAcuteOnly, any non-g1 kappa): fixed truncated-Gamma,
        % handled by the plain 'tgamma' case in the loop above -- NOT
        % part of P.kap, since it no longer participates in the
        % hierarchical density block
        add(kap{j}, 'tgamma', aFix, sFix, 1, Inf, 2.0);
    else
        add(kap{j}, 'hkap', 0, 0, 1, Inf, 2.0);
        kapCols(end + 1, 1) = k; %#ok<AGROW>
        kapGrpKept(end + 1, 1) = remap(g); %#ok<AGROW>
    end
end
% hyperparameters: log-mean and log-shape per KEPT window group only
lmCols = zeros(nnz(keepGrp), 1); laCols = zeros(nnz(keepGrp), 1);
for g = find(keepGrp)'
    add(['kapHyp_lm_' grpNames{g}], 'norm', log(2.5), 0.5, -Inf, Inf, log(2.5));
    lmCols(remap(g)) = k;
end
for g = find(keepGrp)'
    add(['kapHyp_la_' grpNames{g}], 'norm', log(2.0), 0.6, -Inf, Inf, log(2.0));
    laCols(remap(g)) = k;
end

P = struct('params', prm, 'names', {{prm.name}}, 'd', numel(prm));
P.idx = cell2struct(num2cell(1:P.d), P.names, 2);
P.mutateIdx = ~strcmp({prm.type}, 'fixed');
P.kap = struct('kapCols', kapCols, 'groups', kapGrpKept, ...
    'lmCols', lmCols, 'laCols', laCols, 'G', nnz(keepGrp));
end
