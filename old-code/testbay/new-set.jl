using SparseArrays
using Random 
using LinearAlgebra
using DataStructures
using Laplacians

# BLAS implementaion 

struct PositionMap
    p_to_v::Vector{Int} 
    v_to_p::Vector{Int} 
end

function _init_table!(table, A, pmap) 
    fill!(table, (0.0, 1))
    vals = nonzeros(A) 
    for p in size(table, 1) 
        acc = 0.0
        for idx in nzrange(A, pmap.p_to_v[p])
            acc += vals[idx]
        end 
        table[p] = (acc, p)
    end
    return table
end 

tsize(outgoing, _) = outgoing
tops(outgoing, shared) = outgoing + shared

log2sumexp2(x, y) = max(x, y) + log1p(exp2(min(x, y) - max(x, y))) / log(2)

function calc_vals!(between, A, pmap, i, j) 
    outgoing = zero(eltype(A))
    n = size(diag(A))
    x = zeros(n)
    one_vector = ones(n) 
    
    x_ij = zeros(n) 
    x_ij[pmap.p_to_v[i:j]] .= 1
    
    y = A * x_ij

    for k in i;j 
        x[pmap.p_to_v[k]] = 1
        between[k] = dot(x, y .* x) - dot(x, A * x)
    end

    outgoing = dot(x_ij, ((A * one_vector) .* x_ij)) - dot(x_ij, y)
    return outgoing, between
end

function iter_width(A, pmap, cost=tsize, merge=max) 
    n = size(A, 1) 
    table = fill((0.0, 0), (n,n))
    between = zeros(n) 
    return _iter_width!(table, between, A, pmap; cost=cost, merge=merge)
end 

function _iter_width!(table, between, A, pmap; cost=tsize, merge=max) 
    _init_table!(table, A, pmap) 
    n = length(pmap.p_to_v)
    fill!(between, 0) 
    
    for win_size in 2:n 
        for i in 1:(n-win_size -1) 
            j = i + win_size - 1
            outgoing, between = calc_vals!(between, A, pmap, i, j) 
            best = Inf 
            bestk = -1 
            for k in i:(j-1)
                v = pmap.p_to_v[k]
                b = between[k]
                c = cost(outgoing, b) 
                c = merge(merge(c, table[i, k][1]), table[k+1, j][1])

                if c < best
                    best = c 
                    bestk = k
                end
            end
            table[i, j] = (best, bestk) 
        end
    end
    return table 
end

function makeadj(B)
    n = size(B, 2) 
    A = zeros(n, n) 
    for i in axes(B,1)
        nz = findall(x -> x != 0, B[i, :])
        A[nz[1], nz[2]] = 1.0
        A[nz[2], nz[1]] = 1.0
    end
    return A
end

