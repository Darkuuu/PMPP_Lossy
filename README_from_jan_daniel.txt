Project is not finished.

All added code can be found under /src/volren/CUDA_compress.h (everything is written into header).
Small changes were made to /src/volren/RendererRBUC8x8.cu

Code in current state does calculate all four predicator functions on GPU by getting one brick from CPU,
transfering it to GPU, doing ONE predicator function, transfering it back to CPU, encoding it and starting with
the next predicator function.

We also commented everything in the code we would've looked into, if we hadn't mismanage our time so badly.

Missing is following:
- port the encoding on GPU (and calculate one brick completely on GPU)
- distribute the bricks on GPU (potentially with streams to achieve high overlap on memory/computation).
- lossy
- evaluation with nvprof (to check for right memory usage, if memory or computation bound, warp divergence, occupancy, etc.)