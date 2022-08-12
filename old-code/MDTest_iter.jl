using Coarsening, Graphs, LinearOrdering, MatrixMarket, BenchmarkTools

include("./itertreewidth-old.jl")

fname = "../graphs/regular3_32_2_0.mtx"

adj = makeadj(mmread(fname))

G = SimpleGraph(adj);

n = nv(G)
table = fill((0.0, 0), (n,n))
between = zeros(n)

config = (
          compat_sweeps=10,
          stride_percent=0.5,
          gauss_sweeps=10,
          coarsening=VolumeCoarsening(0.4, 2.0, 5),
          coarsest=10,
          pad_percent=0.05,
          node_window_sweeps=10,
          node_window_size=1,
          seed = 0
         )

onesum = PSum(1)

function profileme()
    position_to_idx, idx_to_position = ordergraph(onesum, G; config...);
    pmap = PositionMap(position_to_idx, idx_to_position)
    onesumval = LinearOrdering.evalorder(onesum, adjacency_matrix(G), idx_to_position)
    cost, _ = iter_width(adjacency_matrix(G), pmap)[1, length(position_to_idx)]
end

profileme() 
print("Number of threads: ") 
println(Threads.nthreads())
print("Graph used: ")
println(fname)
@btime profileme()

println()
println()
