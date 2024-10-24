# N-Body Simulation using CUDA

Currently supports ~330k particles on my nvidia RTX 4070 Laptop GPU. Uses the naive `n^2` pair-pair interaction algorithm.

## Preprocessor defines

### Kernel related defines

I recommend setting these to numbers that match well with your GPU's number of multiprocessors and number of threads / multiprocessor.

- `BLOCK_SIZE`: Number of threads per block
- `GRID_SIZE`: Number of blocks

### Simulation related defines

- `N`: number of points
- `GRAVITY`: gravitational constant
- `SOFTENING`: softening used for distance calculation
- `DELTA_T`: timestep
