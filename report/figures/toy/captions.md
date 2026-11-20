# Captions for the toy example figures

- `01_data`: 300 points sampled from a circle with Gaussian noise.
- `02_lens`: The lens assigns each point its height; colour shows the lens value.
- `03_cover`: The range of heights is covered by 6 overlapping intervals (bands).
- `04_clusters`: Points in each band are clustered separately: 4 of the 6 bands split into more than one cluster, giving 10 clusters in total.
- `05_nerve`: Each cluster becomes a node; clusters that share points (in the overlaps) are joined.
- `06_graph`: The resulting Mapper graph has 1 loop, recovering the hole of the circle.
- `07_bad_coarse`: With only two intervals each band holds one arc; their overlap has two pieces, but the nerve records only 1 edge, so the graph has 0 loops.
- `08_bad_no_overlap`: Without overlap no cluster shares points with another, so the nerve has 0 edges and the graph falls apart into 10 pieces with 0 loops.
