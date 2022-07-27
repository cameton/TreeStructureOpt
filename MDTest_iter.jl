using Coarsening, Graphs, LinearOrdering, MatrixMarket, BenchmarkTools, Profile

include("./itertreewidth.jl")
#include("./dynamictreewidth.jl")

#fname = "./graphs/regular3_32_2_0.mtx" #192x192
#fname = "./graphs/regular3_32_2_1.mtx"
#fname = "./graphs/regular3_32_2_2.mtx"
#fname = "./graphs/regular3_32_2_3.mtx"
#fname = "./graphs/regular3_32_2_4.mtx"

#fname = "./graphs/regular4_32_3_0.mtx"
#fname = "./graphs/regular4_32_3_1.mtx"
#fname = "./graphs/regular4_32_3_2.mtx"
#fname = "./graphs/regular4_32_3_3.mtx"
#fname = "./graphs/regular4_32_3_4.mtx"

#fname = "./graphs/regular5_32_4_0.mtx"  #256x256
#fname = "./graphs/regular5_32_4_1.mtx"
#fname = "./graphs/regular5_32_4_2.mtx" 
#fname = "./graphs/regular5_32_4_3.mtx"
fname = "./graphs/regular5_32_4_4.mtx"

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
table = Table(Tuple{Float64, Int}, n)
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
A = adjacency_matrix(G)

# I think the use of global variables might mess with performance, but whatever for rn
function profileme()
    cost, _ = iter_width!(table, between, A, pmap).idata[1][end]
end

profileme() # Do not profile - for precompilation
Profile.clear_malloc_data()
profileme() # Do not profile - for precompilation
# print("Number of threads: ")  
# println(Threads.nthreads())
# print("Graph used: ") 
# println(fname)
# @btime profileme() # Profile this
#@profile profileme()
#Profile.print()
# println()
# println()

