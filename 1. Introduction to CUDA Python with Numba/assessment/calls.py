# Use the 'File' menu above to 'Save' after pasting in your 3 function calls.
%%timeit
# Feel free to modify the 3 function calls in this cell
"""
normalized = normalize(greyscales)
weighted = weigh(normalized, weights)
SOLUTION = activate(weighted)
"""
# transfer inputs to the gpu
greyscales_gpu = cuda.to_device(greyscales)
weights_gpu = cuda.to_device(weights)

# create intermediate arrays and output array on the GPU
normalized_gpu = cuda.device_array(shape=(n,), dtype=np.float32)
weighted_gpu = cuda.device_array(shape=(n,), dtype=np.float32)
activated_gpu = cuda.device_array(shape=(n,), dtype=np.float32)

# calculation
normalized = gpu_normalize(greyscales_gpu, out=normalized_gpu)
weighted = gpu_weigh(normalized_gpu, weights_gpu, out=weighted_gpu)
SOLUTION = gpu_activate(weighted_gpu, out=activated_gpu)