using Coarsening, Graphs, LinearOrdering, MatrixMarket, BenchmarkTools

include("./itertreewidth.jl")
include("./dynamictreewidth.jl")

fname = "./graphs/regular3_32_2_0.mtx" #192x192
#fname = "./graphs/regular5_32_4_2.mtx" #256x256

### New Graphs (Weighted Undirected) ###
#fname = "./graphs/Plants_10NN.mtx" #stackoverflow error 1600x1600
#fname = "./graphs/Binaryalphadigs_10NN.mtx" #Jupyter Kerenel keeps dying 1404x1404
#fname = "./graphs/collins_15NN.mtx" #Jupyter Kernel keeps dying 1000x1000
#fname = "./graphs/Vehicle_10NN.mtx" #Jupyer Kernel keeps dying 846x846
#fname = "./graphs/Ecoli_10NN.mtx" #stackoverflow error 336x336
#fname = "./graphs/YaleA_10NN.mtx" #stackoverdlow error 165x165

adj = makeadj(mmread(fname))

G = SimpleGraph(adj);
n = nv(G)
table = fill((0.0, 0), (n, n))
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

position_to_idx, idx_to_position = ordergraph(onesum, G; config...);
pmap = PositionMap(position_to_idx, idx_to_position)
onesumval = LinearOrdering.evalorder(onesum, adjacency_matrix(G), idx_to_position)

# I think the use of global variables might mess with performance, but whatever for rn
function profileme()
    cost, _ = _iter_width!(table, between, adjacency_matrix(G), pmap)[1, length(position_to_idx)]
end

profileme() # Do not profile - for precompilation
print("Number of threads: ")  
println(Threads.nthreads())
print("Graph used: ") 
println(fname)
@btime profileme() # Profile this

println()
println()

