using SparseArrays
using Random
using LinearAlgebra
using DataStructures
using FLoops

struct PositionMap
    p_to_v::Vector{Int} # position to vertex
    v_to_p::Vector{Int} # vertex to position
end

struct Table{T}
    idata::Vector{Vector{T}}
    jdata::Vector{Vector{T}}
end

function Table(T, n)
    idata = [ Vector{T}(undef, n - i + 1) for i in 1:n ]
    jdata = [ Vector{T}(undef, n - i + 1) for i in n:-1:1 ]
    return Table{T}(idata, jdata)
end

function zerotable!(table)
    for vec in table.idata
        fill!(vec, (0, 0.0))
    end
    for vec in table.jdata
        fill!(vec, (0, 0.0))
    end
end

function _init_table!(table, A, pmap)
    zerotable!(table)
    vals = nonzeros(A)
    for p in axes(A, 2)
        acc = 0.0
        for idx in nzrange(A, pmap.p_to_v[p])
            acc += vals[idx]
        end
        table.idata[p][1] = table.jdata[p][1] = (acc, p)
    end
    return table
end

# Costs
tsize(outgoing, _) = outgoing
tops(outgoing, shared) = outgoing + shared

# Merge
log2sumexp2(x, y) = max(x, y) + log1p(exp2(min(x, y) - max(x, y))) / log(2)

function weighted_degree(A, pmap, i, k, j)
    rows = rowvals(A)
    vals = nonzeros(A)
    left = right = out = zero(eltype(A))
    v = pmap.p_to_v[k]
    for idx in nzrange(A, v)
        r = pmap.v_to_p[rows[idx]]
        if !(i <= r <= j)
            out += vals[idx]
        elseif i <= r < k
            left += vals[idx]
        elseif k < r <= j
            right += vals[idx]
        end
    end
    return out, left, right
end

function calc_vals!(between, A, pmap, i, j) # GPU parallelization on this might be helpful/useful
    outgoing = zero(eltype(A))
    outgoing, _, between[i] = weighted_degree(A, pmap, i, i, j)
    for k in (i + 1):j
        out, left, right = weighted_degree(A, pmap, i, k, j)
        outgoing += out
        between[k] = between[k - 1] + right - left
    end
    return outgoing
end

function iter_width(A, pmap; cost=tsize, merge=max)
    n = size(A, 1)
    table = Table(Tuple{Float64, Int}, n)
    between = zeros(n)
    return _iter_width!(table, between, A, pmap; cost=cost, merge=merge)
end

function find_split(cost, merge, ivec, jvec, between, outgoing, i, j)
    best = Inf; bestk = 0
    for k in i:(j-1)
        b = between[k]

        c = cost(outgoing, b)
        c = merge(merge(c, ivec[k - i + 1][1]), jvec[j - k][1])
        if c < best
            best = c
            bestk = k
        end
    end
    return best, bestk
end

function _iter_width!(table, between, A, pmap, cost, merge)
    for win_size in 2:n
        for shift in 1:win_size
            for i in shift:win_size:(n-win_size+1)
                j = i + win_size - 1
                ivec = table.idata[i] 
                jvec = table.jdata[j]
                outgoing = calc_vals!(between, A, pmap, i, j)

                ivec[win_size] = jvec[win_size] = find_split(cost, merge, ivec, jvec, between, outgoing, i, j)
            end
        end
    end
    return table
end

function iter_width!(table, between, A, pmap; cost=tsize, merge=max)
    _init_table!(table, A, pmap)
    n = length(pmap.p_to_v)
    fill!(between, 0)

    _iter_width!(table, between, A, pmap, cost, merge)

    return table
end

function makeadj(B)
    n = size(B, 2)
    A = zeros(n, n)
    for i in axes(B, 1) 
        nz = findall(x -> x != 0, B[i, :])
        A[nz[1], nz[2]] = 1.0
        A[nz[2], nz[1]] = 1.0 
    end
    return A
end

