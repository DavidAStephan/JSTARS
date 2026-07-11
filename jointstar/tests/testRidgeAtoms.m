classdef testRidgeAtoms < matlab.unittest.TestCase
    %TESTRIDGEATOMS RidgeAtoms block-partition mechanics.
    %
    %   Pure sampler-mechanics check of jointstar.blockPartition (the MH
    %   block-partition builder factored out of jointstar.runSMC): with
    %   RidgeAtoms on, the partition must (i) remain a bijection on 1:dm
    %   -- every mutated column proposed in exactly one block -- and
    %   (ii) never split a named atom across two blocks, across several
    %   independent RNG seeds.  Also checks RidgeAtoms off reproduces the
    %   pre-existing randperm/linspace partition exactly (no behaviour
    %   change unless the option is explicitly requested), and that an
    %   atom with fewer than 2 members present in covCols dissolves
    %   without error.

    methods (Test)

        function partitionValidAndAtomsCohereAcrossSeeds(tc)
            dm = 130;
            covCols = 201:(200 + dm);     % arbitrary absolute column ids
            atomGroups = { ...
                covCols([3, 4]), ...
                covCols([10, 11, 12]), ...
                covCols([50, 51]), ...
                covCols([dm - 1, dm]), ...
                covCols([70, 71, 72, 73])};
            nB = ceil(dm / 40);

            for seed = [1, 2, 3, 42, 999]
                rng(seed);
                [bperm, edgesB] = jointstar.blockPartition(dm, covCols, nB, ...
                    true, atomGroups);

                tc.verifyEqual(sort(bperm), 1:dm, ...
                    sprintf('seed %d: bperm not a permutation of 1:dm', seed));
                tc.verifyEqual(edgesB(1), 0);
                tc.verifyEqual(edgesB(end), dm);
                tc.verifyTrue(all(diff(edgesB) >= 0), ...
                    sprintf('seed %d: block edges not monotone', seed));

                % block index carried by each bperm slot, then inverted
                % back onto covCols positions
                blockOf = zeros(1, dm);
                for b = 1:nB
                    blockOf(edgesB(b) + 1:edgesB(b + 1)) = b;
                end
                posBlock = zeros(1, dm);
                posBlock(bperm) = blockOf;

                for a = 1:numel(atomGroups)
                    pos = find(ismember(covCols, atomGroups{a}));
                    blocks = unique(posBlock(pos));
                    tc.verifyEqual(numel(blocks), 1, sprintf( ...
                        'seed %d atom %d: members split across blocks %s', ...
                        seed, a, mat2str(blocks)));
                end
            end
        end

        function ridgeAtomsOffMatchesClassicPartition(tc)
            dm = 84; covCols = 1:dm; nB = 3;
            atomGroups = {covCols([1, 2]), covCols([40, 41, 42])};

            rng(7);
            [bperm1, edges1] = jointstar.blockPartition(dm, covCols, nB, false, atomGroups);
            rng(7);
            bperm2 = randperm(dm);
            edges2 = round(linspace(0, dm, nB + 1));

            tc.verifyEqual(bperm1, bperm2, ...
                'RidgeAtoms=false must reproduce randperm(dm) exactly');
            tc.verifyEqual(edges1, edges2);

            % omitting the ridgeAtoms/atomGroups args entirely must behave
            % identically to explicitly passing false
            rng(7);
            [bperm3, edges3] = jointstar.blockPartition(dm, covCols, nB);
            tc.verifyEqual(bperm3, bperm2);
            tc.verifyEqual(edges3, edges2);
        end

        function smallAndAbsentAtomsDissolveGracefully(tc)
            dm = 20; covCols = 1:dm;
            atomGroups = { ...
                5, ...              % single column: never a valid atom
                [100, 101], ...     % wholly absent from covCols
                [7, 8, 9]};         % a real 3-column atom
            rng(1);
            [bperm, edgesB] = jointstar.blockPartition(dm, covCols, 2, true, atomGroups);
            tc.verifyEqual(sort(bperm), 1:dm);

            blockOf = zeros(1, dm);
            for b = 1:2
                blockOf(edgesB(b) + 1:edgesB(b + 1)) = b;
            end
            posBlock = zeros(1, dm);
            posBlock(bperm) = blockOf;
            tc.verifyEqual(numel(unique(posBlock([7, 8, 9]))), 1, ...
                'the surviving 3-column atom must stay in one block');
        end

    end
end
