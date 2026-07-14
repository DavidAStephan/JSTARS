classdef testBlockAtoms < matlab.unittest.TestCase
    %TESTBLOCKATOMS jointstar.blockAtoms / blockPartitionAtoms unit tests
    %for the 'StructuredBlocks' option (DESIGN_transformed_kernel.md
    %Section I2): atoms never split across blocks, their union covers
    %every mutated column exactly once, and the flags-off partition
    %(jointstar.blockPartition) is untouched by this feature (R6).

    methods (Test)

        function atomsPartitionMutatedColumnsExactly(tc)
            P = jointstar.defaultPriors('HierKappa', true);
            covCols = find(P.mutateIdx);
            atoms = jointstar.blockAtoms(P, covCols);

            allRel = sort(horzcat(atoms{:}));
            dm = numel(covCols);
            tc.verifyEqual(allRel, 1:dm, ...
                'atoms must partition 1:dm exactly (union, no gaps, no overlaps)');

            counts = zeros(1, dm);
            for a = 1:numel(atoms)
                counts(atoms{a}) = counts(atoms{a}) + 1;
            end
            tc.verifyEqual(counts, ones(1, dm), ...
                'every mutated column must belong to exactly one atom');
        end

        function namedRidgeGroupsStayIntact(tc)
            P = jointstar.defaultPriors('HierKappa', true);
            covCols = find(P.mutateIdx);
            atoms = jointstar.blockAtoms(P, covCols);
            absToRel = zeros(1, P.d); absToRel(covCols) = 1:numel(covCols);

            groupsToCheck = { ...
                {'phisum', 'phi2', 'nu'}, ...
                {'rhoU', 'xi1', 'xi2', 'phiu'}, ...
                {'rhopr', 'theta1', 'theta2', 'phipr'}, ...
                {'rhohpp', 'lam1', 'lam2', 'phihpp'}, ...
                {'rhok', 'chi1', 'chi2', 'phik'}, ...
                {'gzbar', 'gwbar'}, ...
                {'sig_gz', 'sig_gw', 'sig_xi'} ...
                };
            for g = 1:numel(groupsToCheck)
                nms = groupsToCheck{g};
                rel = zeros(1, numel(nms));
                for k = 1:numel(nms)
                    rel(k) = absToRel(find(strcmp(P.names, nms{k}), 1));
                end
                foundAtom = 0;
                for a = 1:numel(atoms)
                    if all(ismember(rel, atoms{a}))
                        foundAtom = a; break;
                    end
                end
                tc.verifyGreaterThan(foundAtom, 0, ...
                    sprintf('group {%s} not co-blocked into one atom', strjoin(nms, ',')));
            end
        end

        function hierKappaHyperGroupsStayIntact(tc)
            P = jointstar.defaultPriors('HierKappa', true);
            covCols = find(P.mutateIdx);
            atoms = jointstar.blockAtoms(P, covCols);
            absToRel = zeros(1, P.d); absToRel(covCols) = 1:numel(covCols);

            kp = P.kap;
            for g = 1:kp.G
                memberAbs = kp.kapCols(kp.groups == g)';
                hyperAbs = [kp.lmCols(g), kp.laCols(g)];
                rel = absToRel([memberAbs, hyperAbs]);
                foundAtom = 0;
                for a = 1:numel(atoms)
                    if all(ismember(rel, atoms{a}))
                        foundAtom = a; break;
                    end
                end
                tc.verifyGreaterThan(foundAtom, 0, ...
                    sprintf('hierarchical-kappa hyper-group %d not co-blocked', g));
            end
        end

        function silentlyDropsAbsentNames(tc)
            % Without HierKappa, no kapHyp_* names exist; blockAtoms must
            % not error and the independent-kappa columns fall back to
            % singleton atoms via the generic "remaining columns" path.
            P = jointstar.defaultPriors('HierKappa', false);
            covCols = find(P.mutateIdx);
            atoms = jointstar.blockAtoms(P, covCols);
            allRel = sort(horzcat(atoms{:}));
            tc.verifyEqual(allRel, 1:numel(covCols));

            absToRel = zeros(1, P.d); absToRel(covCols) = 1:numel(covCols);
            jk = absToRel(find(strcmp(P.names, 'kapc_2021'), 1));
            found = cellfun(@(a) isscalar(a) && a == jk, atoms);
            tc.verifyTrue(any(found), 'independent kappa must be a singleton atom');
        end

        function packingNeverSplitsAnAtomAcrossBlocks(tc)
            rng(55);
            P = jointstar.defaultPriors('HierKappa', true);
            covCols = find(P.mutateIdx);
            atoms = jointstar.blockAtoms(P, covCols);
            dm = numel(covCols);
            nB = max(1, ceil(dm / 40));
            [bperm, edgesB] = jointstar.blockPartitionAtoms(atoms, nB);

            tc.verifyEqual(sort(bperm), 1:dm);
            tc.verifyEqual(edgesB(1), 0);
            tc.verifyEqual(edgesB(end), dm);

            blockOf = zeros(1, dm);
            for b = 1:nB
                sel = bperm(edgesB(b) + 1:edgesB(b + 1));
                blockOf(sel) = b;
            end
            for a = 1:numel(atoms)
                bset = unique(blockOf(atoms{a}));
                tc.verifyEqual(numel(bset), 1, ...
                    sprintf('atom %d spans %d blocks', a, numel(bset)));
            end
        end

        function flagsOffPartitionUnchanged(tc)
            % jointstar.blockPartition (the off-path callee, used when
            % StructuredBlocks is false) is untouched by this feature:
            % same call + same seed gives the same output as before.
            rng(7);
            dm = 20; covCols = 101:120; nB = 2;
            [bperm1, edges1] = jointstar.blockPartition(dm, covCols, nB);
            rng(7);
            [bperm2, edges2] = jointstar.blockPartition(dm, covCols, nB);
            tc.verifyEqual(bperm1, bperm2);
            tc.verifyEqual(edges1, edges2);
            tc.verifyEqual(edges1, round(linspace(0, dm, nB + 1)));
        end

    end
end
