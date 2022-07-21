using Graphs
using SparseArrays
using CuthillMcKee: symrcm
using Random
using QXGraphDecompositions: flow_cutter
using Combinatorics
using ProgressMeter
using LinearAlgebra
using KrylovKit
using DataStructures
using LRUCache
using LinearOrdering 
using Coarsening

import LightGraphs

struct Leaf
    idx::Int 
    v::Int
end

struct Node
    lo::Int
    up::Int 
    left::Int
    right::Int
end 

struct TreeDecomposition
    lo::Int
    up::Int
    width::Int
    left::Union{TreeDecomposition, Nothing}
    right::Union{TreeDecomposition, Nothing} 
    bag::Set{Graphs.SimpleGraphs.SimpleEdge{Int64}}
end 

struct NewTreeDecomposition
    tree::Vector{Union{Leaf, Node}}
    bags::Vector{SortedSet{Int}}
    widths::Vector{Float64}
end

function NewTreeDecomposition(n)
    tree = Vector{Union{Leaf, Node}}(undef, 2 * n -1)
    bags = Vector{SortedSet{Int}}(undef, 2 * n - 1)
    widths = zeros(2 * n - 1)
    return NewTreeDecomposition(tree, bags, widths)
end


function vectorize_decomp!(bags, tree, parent_idx, td)
    if isnothing(td)
        return bags, tree
    end
    idx = length(bags) + 1
    push!(bags, td.bag)
    add_vertex!(tree)
    add_edge!(tree, parent_idx, idx)
    vectorize_decomp!(bags, tree, idx, td.left)
    vectorize_decomp!(bags, tree, idx, td.right)
    return bags, tree
end

function vectorize_decomp(td)
    bags = [td.bag]
    tree = SimpleGraph(1)
    vectorize_decomp!(bags, tree, 1, td.left)
    return vectorize_decomp!(bags, tree, 1, td.right)
end


function check_td(bags, tree, g)
    bigbag = union(bags...)
    edgemap = Dict()
    incident = []
    for v in vertices(g)
        push!(incident, Graphs.SimpleGraphs.SimpleEdge{Int64}[])
    end

    # Node Coverage
    nc = true
    for (i, e) in enumerate(edges(g))
        edgemap[e] = i
        nc = nc && (e ∈ bigbag)
        push!(incident[e.src], e)
        push!(incident[e.dst], e)
    end

    # Edge Coverage
    adjmap = spzeros(ne(g), ne(g))
    ec = true
    for bag in bags
        for e1 in bag
            for e2 in bag
                adjmap[edgemap[e1], edgemap[e2]] = 1.0
            end
        end
    end
    for v in vertices(g)
        for e1 in incident[v]
            for e2 in incident[v]
                ec = ec && (adjmap[edgemap[e1], edgemap[e2]] == 1.0)
            end
        end
    end

    # Coherence
    co = true
    for (i, bag1) in enumerate(bags)
        for (j, bag2) in enumerate(bags)
            inter = intersect(bag1, bag2)
            pos = i
            for e in a_star(tree, i, j)
                pos = ifelse(pos == e.src, e.dst, e.src)
                co = co && issubset(inter, bags[pos])
            end
        end
    end
    return nc && ec && co, (nc, ec, co)
end

function oob(vertex_to_position, i, j, e)
    a = vertex_to_position[e.src]
    b = vertex_to_position[e.dst]
    return (i <= a <= j) ⊻ (i <= b <= j)
end

function outgoing_edges(G, incident_edges, position_to_vertex, vertex_to_position, i, j)
    inc = Iterators.flatten((v -> incident_edges[v]).(position_to_vertex[i:j]))
    return Set(Iterators.filter(e -> oob(vertex_to_position, i, j, e), inc))
end


vertex_pair(v, u) = Edge(min(v, u), max(v, u))
incident_edges(G, v) = (vertex_pair(v, u) for u in neighbors(G, u))

function _weight_to_intervals(A, position_to_vertex, vertex_to_position, interval, to_intervals)
    acc = zero(eltype(A))
    #Here allocations go up making speed signifcantly decrease
    #Threads.@threads for c in interval
    for c in interval
        acc += _weight_to_intervals(A, vertex_to_position, position_to_vertex[c], to_intervals)
    end
    return acc
end
function _weight_to_intervals(A, vertex_to_position, v, to_intervals)
    rows = rowvals(A)
    vals = nonzeros(A)
    acc = zero(eltype(A))
    for idx in nzrange(A, v)
        r = vertex_to_position[rows[idx]]
        if any(r in to_interval for to_interval in to_intervals)
            acc += vals[idx]
        end
    end
    return acc
end

myfindmin(f, domain) = mapfoldl( v -> (f(v), v), _my_rf_findmin, domain )
_my_rf_findmin(a, b) = ifelse(Base.isgreater(a[1], b[1]), b, a)

function init_table!(f, table, idx...)
    if isnothing(table[idx...]) # Check for efficiency sometime; maybe use a macro and cartesian indices
        table[idx...] = f()
    end
    return table[idx...]
end

function calcbetween(A, position_to_vertex, vertex_to_position, i, j)
    between = zeros(j - i + 1)
    for l in 1:(j - i)
        k = i + l - 1
        v = position_to_vertex[k]
        between[l] += _weight_to_intervals(A, vertex_to_position, v, ((k+1):j,))
        between[l] -= _weight_to_intervals(A, vertex_to_position, v, (i:k,))
        between[l + 1] = between[l]
    end
    between[end] = _weight_to_intervals(A, vertex_to_position, position_to_vertex[j], (i:(j-1),))
    return between
end

### Not acutally recursion, just keeping the name for conventioni

cache = LRU{Tuple{Int, Int}, Tuple{eltype(A), Int}}(; maxsize=maxsize, by=sizeof)

function _recursive_width!(cache, A, position_to_vertex, vertex_to_position, i, j; carving = false, flops = false) 
    

    if i == j
        c = Coarsening.sumcol(A, position_to_vertex[i])
        if flops
            return 2.0 ^ c, i
        end
        return c, i
    end

    n = length(position_to_vertex)
    outgoing = _weight_to_intervals(A, position_to_vertex, vertex_to_position, i:j, (1:(i-1), (j+1):n))
    best = Inf
    bestk = -1
    between = calcbetween(A, position_to_vertex, vertex_to_position, i, j)

    #Threads.@threads for k in i:(j-1)
    for k in i:(j-1)
        v = position_to_vertex[k]
        b = between[k - i + 1]

        if !carving
            b + outgoing < best ? nothing : continue
        end 

       #= # What to do with this 
        (l, _) = get!(cache, (i, k)) do 
            _recursive_width!(cache, A, position_to_vertex, vertex_to_position, i, k; carving=carving, flops = flops)
        end 
=#
        Threads.@threads for left_node in i:k
            
            (l, _) = get!(cache, (i, k))

            if i == k 
                c = Coarsening.sumcol(A, position_to_vertex[i])
                if flops 
                    return 2.0 ^ c, i
                end
                return c, i
            end 

            v = position_to_vertex[left_node]
            b = between[left_node - i + 1] 
            
            
            if !carving 
                b + outgoing < best ? nothing : continue 
            end 
            
            if flops 
                2.0 ^ (b + outgoing) + l < best ? nothing : continue
            end 

            l < best ? nothing : continue 
            
            Threads.@threads for right_node in (k+1):j 
                
                (r, _) = get!(cache, (k+1, j))

                if (k+1) == j 
                    c = Coarsening.sumcol(A, position_to_vertex[k+1])
                    if flops 
                        return 2.0^ c, (k+1)
                    end
                    return c, (k+1)
                end 

                v = position_to_vertex[right_node]
                b = between[right_node - k + 2]
            
                if !carving 
                    b + outgoing < best ? nothing : continue
                end 

                if flops 
                    2.0 ^ (b + outgoing) + l < best ? nothing : continue 
                end 

                l < best ? nothing : continue
            end
        end 
        #=
        if flops 
            2.0 ^ (b + outgoing) + l < best ? nothing : continue 
        end 

        l < best ? nothing : continue
        
        # What to do with this 
        (r, _) = get!(cache, (k+1, j)) do 
            _recursive_width!(cache, A, position_to_vertex, vertex_to_position, k+1, j; carving = carving, flops = flops)
        end
=#
        if carving
            testval = max(outgoing, l, r)
        elseif flops
            testval =2.0 ^ (outgoing +b) +l + r
        else 
            testval = max(b + outgoing, l, r)
        end 

        if testval < best 
            best = testval
            bestk = k
        end
    end
    return best, bestk
### Recursion --> switch to not recurision.
end 

function recursive_width(A, position_to_vertex, vertex_to_position, i=1, j=size(A, 1); maxsize=2^28, flops=false, carving=false)
    cache = LRU{Tuple{Int, Int}, Tuple{Float64, Int}}(; maxsize=maxsize, by=sizeof)
    return _recursive_width!(cache, A, position_to_vertex, vertex_to_position, i, j; flops=flops, carving=carving)
end

function recursive_width_td!(cache, td, idx, A, position_to_vertex, vertex_to_position, i, j)
    if i == j
    end
end

function recursive_width_td(A, position_to_vertex, vertex_to_position, i=1, j=size(A, 1); maxsize=2^28)
    cache = LRU{Tuple{Int, Int}, Tuple{eltype(A), Int}}(; maxsize=maxsize, by=sizeof)
    td = NewTreeDecomposition(length(position_to_vertex))
    Q = PriorityQueue{UnitRange{Int}, Int}()
    return _recursive_width_td!(cache, td, A, position_to_vertex, vertex_to_position, i, j)
end

# order: position -> vertex
function recursion_opt(cache, G, incident_edges, position_to_vertex, vertex_to_position, i, j)
    if i == j
        inc = Set(incident_edges[position_to_vertex[i]])
        return TreeDecomposition(i, j, length(inc), nothing, nothing, inc)
    end
    outgoing = outgoing_edges(G, incident_edges, position_to_vertex, vertex_to_position, i, j)
    best_width = Inf # make type stable
    best_bag = Set()
    best_left = nothing
    best_right = nothing
    #Adding threads here does cause speed up
    Threads.@threads for k in i:(j-1)
    #for k in i:(j-1)
        left_tree = get!(cache, (i, k)) do
            recursion_opt(cache, G, incident_edges, position_to_vertex, vertex_to_position, i, k)
        end
        right_tree = get!(cache, (k+1, j)) do
            recursion_opt(cache, G, incident_edges, position_to_vertex, vertex_to_position, k+1, j)
        end
        bag = union(outgoing, intersect(left_tree.bag, right_tree.bag))
        width = max(length(bag), left_tree.width, right_tree.width)
        if width < best_width
            best_width = width
            best_bag = bag
            best_left = left_tree
            best_right = right_tree
        end
    end
    return TreeDecomposition(i, j, best_width, best_left, best_right, best_bag)
end

function order_width(G, position_to_vertex, vertex_to_position)
    cache = LRU{Tuple{Int, Int}, TreeDecomposition}(; maxsize=2^28, by=sizeof)
    incident_edges = []
    # Threads slow down here
    #Threads.@threads for v in vertices(G)
    for v in vertices(G)
        push!(incident_edges, Graphs.SimpleGraphs.SimpleEdge{Int64}[])
    end
    # Threads slow down here
    #Threads.@threads for e in edges(G)
    for e in edges(G)
        push!(incident_edges[e.src], e)
        push!(incident_edges[e.dst], e)
    end
    tree = recursion_opt(cache, G, incident_edges, position_to_vertex, vertex_to_position, 1, nv(G))
    empty!(cache)
    return tree
end


function makeadj(B)
    n = size(B, 2)
    A = zeros(n, n)
    #Threads here cause a slwo down here
    for i in axes(B, 1)
        nz = findall(x -> x != 0, B[i, :])
        A[nz[1], nz[2]] = 1.0
        A[nz[2], nz[1]] = 1.0
    end
    return A
end

