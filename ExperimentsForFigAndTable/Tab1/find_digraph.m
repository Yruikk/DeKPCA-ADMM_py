function [digraph] = find_digraph(undigraph)
[n,~] = size(undigraph);
tmp = sum(undigraph);
digraph = eye(n);


[~, idx] = min(tmp);
record=[];
start = idx;
for ii = 1: n
    record = [record; idx];
    min_d = inf;
    min_idx = 0;
    for iter = 1: n
        if iter ~= idx && iter ~= start && undigraph(idx, iter) == 1 && sum(digraph(:,iter)) == 1 && sum(digraph(iter,:)) == 1 && tmp(iter)<min_d
            min_d = tmp(iter);
            min_idx = iter;
        end
    end
    if min_idx
        digraph(idx, min_idx) = 1;
        idx = min_idx;
    else
        if ~undigraph(idx, start)
            fprintf('fail\n');
            break;
        else
            digraph(idx, start) = 1;  
            record=[record; start];
            fprintf('success\n');
        end
    end
end
record
end