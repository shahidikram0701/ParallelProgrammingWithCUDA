## Parallel Scan

### Objective

The purpose of this lab is to implement one or more kernels and their associated host code to perform parallel scan on a 1D list. The scan operator used 
will be addition. You should implement the work- efficient kernel discussed in lecture. Your kernel should be able to handle input lists of arbitrary length. 
However, for simplicity, you can assume that the input list will be at most 2,048 * 2,048 elements.

### Instruction

The boundary condition can be handled by filling 'identity value (0 for sum)' into the shared memory of the last block when the length is not a multiple of 
the thread block size.