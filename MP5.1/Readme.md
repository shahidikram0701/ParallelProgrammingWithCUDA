# List Reduction

## Objective

Implement a kernel and associated host code that performs reduction of a 1D list stored in a C array. The reduction should give the sum of the list. You should implement the improved kernel discussed in the lecture. Your kernel should be able to handle input lists of arbitrary length.

## Instruction

The input list will contain at most 2048 x 65535 elements so that it can be handled by only one kernel launch. The boundary condition can be handled by filling ‘identity value (0 for sum)’ into the shared memory of the last block when the length is not a multiple of the thread block size. Write a host (CPU) loop to calculate the total of the reduction sums of each section generated by individual blocks.