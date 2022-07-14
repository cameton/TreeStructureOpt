using Coarsening, Graphs, LinearOrdering, MatrixMarket

include("./dynamictreewidth.jl")

fname = "./graphs/regular3_32_2_0.mtx"
adj = makeadj(mmread(fname))
G = SimpleGraph(adj);
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

# I think the use of global variables might mess with performance, but whatever for rn
function profileme()
    position_to_idx, idx_to_position = ordergraph(onesum, G; config...);
    onesumval = LinearOrdering.evalorder(onesum, adjacency_matrix(G), idx_to_position)
    cost, _ = recursive_width(adjacency_matrix(G), position_to_idx, idx_to_position; flops=true, carving=false)
end

profileme() # Do not profile - for precompilation
println("START")
profileme() # Profile this
println("END")
