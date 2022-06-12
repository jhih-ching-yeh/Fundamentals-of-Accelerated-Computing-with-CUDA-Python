# GPU-Accelerated-Computing-with-Python
NVIDIA’s CUDA Python provides a driver and runtime API for existing toolkits and libraries to simplify GPU-based accelerated processing. Python is one of the most popular programming languages for science, engineering, data analytics, and deep learning applications. However, as an interpreted language, it’s been considered too slow for high-performance computing.

## 1. Introduction to CUDA Python with Numba
The CUDA compute platform enables remarkable application acceleration by enabling developers to execute code in a massively parallel fashion on NVIDA GPUs.
Numba is a just-in-time Python function compiler that exposes a simple interface for accelerating numerically-focused Python functions. 

Numba is a very attractive option for Python programmers wishing to GPU accelerate their applications without needing to write C/C++ code, especially for developers already performing computationally heavy operations on NumPy arrays. Numba can be used to accelerate Python functions for the CPU, as well as for NVIDIA GPUs. The focus of this course is the fundamental techniques needed to GPU-accelerate Python applications using Numba.

## 2. Custom CUDA Kernels in Python with Numba
In this section we will go further into our understanding of how the CUDA programming model organizes parallel work, and will leverage this understanding to write custom CUDA kernels, functions which run in parallel on CUDA GPUs. Custom CUDA kernels, in utilizing the CUDA programming model, require more work to implement than, for example, simply decorating a ufunc with @vectorize. However, they make possible parallel computing in places where ufuncs are just not able, and provide a flexibility that can lead to the highest level of performance.

This section contains three appendices for those of you interested in futher study: a variety of debugging techniques to assist your GPU programming, links to CUDA programming references, and coverage of Numba supported random number generation on the GPU.

## 3. Multidimensional Grids and Shared Memory for CUDA Python with Numba
Now that you can write correct CUDA kernels, and understand the importance of launching grids that give the GPU sufficient opportunity to hide latency, you are going to learn techniques to effectively utilize GPU memory subsystems. These techniques are widely applicable to a variety of CUDA applications, and some of the most important when it comes time to make your CUDA code go fast.

You are going to begin by learning about memory coalescing. To challenge your ability to reason about memory coalescing, and to expose important details relevent to many CUDA applications, you will then learn about 2-dimensional grids and thread blocks. Next you will learn about a very fast, user-controlled, on-demand memory space called shared memory, and will use shared memory to facilitate memory coalescing where it would not have otherwise been possible. Finally, you will learn about shared memory bank conflicts, which can spoil the performance possibilities of using shared memory, and a technique to address them.
