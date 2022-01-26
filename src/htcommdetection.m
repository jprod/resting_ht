function htcommdetection(pwd)
corrvox = readNPY(pwd + "\avgcorrvoxfisher.npy");

commat = zeros(100, size(corrvox, 1));
disp(size(commat));
for i = 1:size(commat,1)
    disp("process user: " + i);
    commbuffer = get4comms();
    commat(i,:) = commbuffer;
end

disp(size(commat));
writeNPY(commat, 'commat100run.npy');

corrmatlvl2 = corrcoef(commat);
[M2,Q2] = community_louvain(corrmatlvl2,[],[],'negative_asym');
writeNPY(M2, 'commlvl2.npy');

function comms = get4comms()
    [M,Q] = community_louvain(corrvox,[],[],'negative_asym');
    MaxM = max(M);
    if MaxM ~= 4
        comms = get4comms();
    else
        comms = M;
    end
end
end