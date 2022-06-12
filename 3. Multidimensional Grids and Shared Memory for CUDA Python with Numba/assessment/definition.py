# Use the 'File' menu above to 'Save' after pasting in your own mm_shared function definition.
import numpy as np
from numba import cuda, types

    
@cuda.jit
def mm_shared(a, b, c):
    sum = 0

    # `a_cache` and `b_cache` are already correctly defined
    a_cache = cuda.shared.array(block_size, types.int32)
    b_cache = cuda.shared.array(block_size, types.int32)

    # TODO: use each thread to populate one element each a_cache and b_cache
    x,y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x
    TPB = int(N)
    
    for i in range(a.shape[1] / TPB):
        a_cache[tx, ty] = a[x, ty + i * TPB]
        b_cache[tx, ty] = b[tx + i * TPB, y]
    
    cuda.syncthreads()
    for j in range(TPB):#a.shape[1]):
        # TODO: calculate the `sum` value correctly using values from the cache 
        sum += a_cache[tx][j] * b_cache[j][ty]
    cuda.syncthreads()    
    c[x][y] = sum