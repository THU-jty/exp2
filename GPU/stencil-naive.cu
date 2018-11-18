#include "common.h"

const char* version_name = "A naive base-line";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

__global__ void stencil_7_naive_kernel_1step(cptr_t in, ptr_t out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    if(tx < nx && ty < ny && tz < nz) {
        int ldx = nx + halo_x * 2;
        int ldy = ny + halo_y * 2;
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;
        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * in[INDEX(x, y, z, ldx, ldy)] \
            + ALPHA_NZZ * in[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PZZ * in[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_ZNZ * in[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_ZPZ * in[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_ZZN * in[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_ZZP * in[INDEX(x, y, z+1, ldx, ldy)];
    }
}

__global__ void stencil_27_naive_kernel_1step(cptr_t in, ptr_t out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    if(tx < nx && ty < ny && tz < nz) {
        int ldx = nx + halo_x * 2;
        int ldy = ny + halo_y * 2;
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;
        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * in[INDEX(x, y, z, ldx, ldy)] \
            + ALPHA_NZZ * in[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PZZ * in[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_ZNZ * in[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_ZPZ * in[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_ZZN * in[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_ZZP * in[INDEX(x, y, z+1, ldx, ldy)] \
            + ALPHA_NNZ * in[INDEX(x-1, y-1, z, ldx, ldy)] \
            + ALPHA_PNZ * in[INDEX(x+1, y-1, z, ldx, ldy)] \
            + ALPHA_NPZ * in[INDEX(x-1, y+1, z, ldx, ldy)] \
            + ALPHA_PPZ * in[INDEX(x+1, y+1, z, ldx, ldy)] \
            + ALPHA_NZN * in[INDEX(x-1, y, z-1, ldx, ldy)] \
            + ALPHA_PZN * in[INDEX(x+1, y, z-1, ldx, ldy)] \
            + ALPHA_NZP * in[INDEX(x-1, y, z+1, ldx, ldy)] \
            + ALPHA_PZP * in[INDEX(x+1, y, z+1, ldx, ldy)] \
            + ALPHA_ZNN * in[INDEX(x, y-1, z-1, ldx, ldy)] \
            + ALPHA_ZPN * in[INDEX(x, y+1, z-1, ldx, ldy)] \
            + ALPHA_ZNP * in[INDEX(x, y-1, z+1, ldx, ldy)] \
            + ALPHA_ZPP * in[INDEX(x, y+1, z+1, ldx, ldy)] \
            + ALPHA_NNN * in[INDEX(x-1, y-1, z-1, ldx, ldy)] \
            + ALPHA_PNN * in[INDEX(x+1, y-1, z-1, ldx, ldy)] \
            + ALPHA_NPN * in[INDEX(x-1, y+1, z-1, ldx, ldy)] \
            + ALPHA_PPN * in[INDEX(x+1, y+1, z-1, ldx, ldy)] \
            + ALPHA_NNP * in[INDEX(x-1, y-1, z+1, ldx, ldy)] \
            + ALPHA_PNP * in[INDEX(x+1, y-1, z+1, ldx, ldy)] \
            + ALPHA_NPP * in[INDEX(x-1, y+1, z+1, ldx, ldy)] \
            + ALPHA_PPP * in[INDEX(x+1, y+1, z+1, ldx, ldy)];
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

#define BLOCK_SIZE 9

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size (ceiling(nx, BLOCK_SIZE), ceiling(ny, BLOCK_SIZE), ceiling(nz, BLOCK_SIZE));
    dim3 block_size (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    for(int t = 0; t < nt; ++t) {
        stencil_7_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size (ceiling(nx, BLOCK_SIZE), ceiling(ny, BLOCK_SIZE), ceiling(nz, BLOCK_SIZE));
    dim3 block_size (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    for(int t = 0; t < nt; ++t) {
        stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}