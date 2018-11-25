#include <stdio.h>
#include <stdlib.h>
#include "common.h"

const char* version_name = "Optimized version";

const int halo = 1;

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    grid_info->halo_size_x = halo;
    grid_info->halo_size_y = halo;
    grid_info->halo_size_z = halo;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

#define XX 32
#define YY 4
#define ZZ 4
//#define lenx (XX+16*2)
#define lenx (XX+halo*2)
#define leny (YY+halo*2)
#define lenz (ZZ+halo*2)
#define BLOCK_SIZE 9

#define __HALO
//(32,4,4)

__global__ void stencil_7_naive_kernel_1step(cptr_t in, ptr_t out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    if(tx < nx && ty < ny && tz < nz) {
        int ldx = nx + halo_x * 2;
        int ldy = ny + halo_y * 2;
        int ldz = nz + halo_z * 2;
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;
#ifdef __HALO		
		__shared__ double sub[lenx*leny*lenz];
		
		//int xx = threadIdx.x+16;
		int xx = threadIdx.x+halo;
		int yy = threadIdx.y+halo;
		int zz = threadIdx.z+halo;
		
		sub[ INDEX( xx, yy, zz, lenx, leny ) ] = in[ INDEX(x, y, z, ldx, ldy) ];
		// if( xx == 16 ) 	sub[ INDEX( xx-1, yy, zz, lenx, leny ) ] = in[ INDEX(x-1, y, z, ldx, ldy) ];
		// if( xx == XX+15 ) 	sub[ INDEX( xx+1, yy, zz, lenx, leny ) ] = in[ INDEX(x+1, y, z, ldx, ldy) ];
		
		if( xx == 1 ) 	sub[ INDEX( xx-1, yy, zz, lenx, leny ) ] = in[ INDEX(x-1, y, z, ldx, ldy) ];
		if( xx == XX ) 	sub[ INDEX( xx+1, yy, zz, lenx, leny ) ] = in[ INDEX(x+1, y, z, ldx, ldy) ];
		if( yy == 1 )	sub[ INDEX( xx, yy-1, zz, lenx, leny ) ] = in[ INDEX(x, y-1, z, ldx, ldy) ];
		if( yy == YY ) 	sub[ INDEX( xx, yy+1, zz, lenx, leny ) ] = in[ INDEX(x, y+1, z, ldx, ldy) ];
		if( zz == 1 )	sub[ INDEX( xx, yy, zz-1, lenx, leny ) ] = in[ INDEX(x, y, z-1, ldx, ldy) ];
		if( zz == ZZ ) 	sub[ INDEX( xx, yy, zz+1, lenx, leny ) ] = in[ INDEX(x, y, z+1, ldx, ldy) ];	
		
		__syncthreads(); 
		
        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * sub[ INDEX( xx, yy, zz, lenx, leny ) ] \
            + ALPHA_NZZ * sub[ INDEX( xx-1, yy, zz, lenx, leny ) ] \
            + ALPHA_PZZ * sub[ INDEX( xx+1, yy, zz, lenx, leny ) ] \
            + ALPHA_ZNZ * sub[ INDEX( xx, yy-1, zz, lenx, leny ) ] \
            + ALPHA_ZPZ * sub[ INDEX( xx, yy+1, zz, lenx, leny ) ] \
            + ALPHA_ZZN * sub[ INDEX( xx, yy, zz-1, lenx, leny ) ] \
            + ALPHA_ZZP * sub[ INDEX( xx, yy, zz+1, lenx, leny ) ];
			
#else
		__shared__ double sub[XX*YY*ZZ];
		
		int xx = threadIdx.x;
		int yy = threadIdx.y;
		int zz = threadIdx.z;
		sub[ INDEX( xx, yy, zz, XX, YY ) ] = in[ INDEX(x, y, z, ldx, ldy) ];
		
		__syncthreads(); 
		
        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * sub[ INDEX( xx, yy, zz, XX, YY ) ] \
			+ ALPHA_NZZ * in[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PZZ * in[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_ZNZ * in[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_ZPZ * in[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_ZZN * in[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_ZZP * in[INDEX(x, y, z+1, ldx, ldy)] ;

#endif
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size (ceiling(nx, XX), ceiling(ny, YY), ceiling(nz, ZZ));
    dim3 block_size (XX, YY, ZZ);
    for(int t = 0; t < nt; ++t) {
        stencil_7_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
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
		__shared__ double sub[lenx*leny*lenz];
		
		int xx = threadIdx.x+halo;
		int yy = threadIdx.y+halo;
		int zz = threadIdx.z+halo;
		int i, j, k;
		if( xx == 1 ) i = -1;
		else if( xx == XX ) i = 1;
		else i = 0;
		
		if( yy == 1 ) j = -1;
		else if( yy == YY ) j = 1;
		else j = 0;
		
		if( zz == 1 ) k = -1;
		else if( zz == ZZ ) k = 1;
		else k = 0;
		
		sub[ INDEX( xx, yy, zz, lenx, leny ) ] = in[ INDEX(x, y, z, ldx, ldy) ];
		if( i != 0 ) 			sub[ INDEX( xx+i, yy, zz, lenx, leny ) ] = in[ INDEX(x+i, y, z, ldx, ldy) ];
		if( j != 0 ) 			sub[ INDEX( xx, yy+j, zz, lenx, leny ) ] = in[ INDEX(x, y+j, z, ldx, ldy) ];
		if( k != 0 )			sub[ INDEX( xx, yy, zz+k, lenx, leny ) ] = in[ INDEX(x, y, z+k, ldx, ldy) ];
		if( i != 0 && j != 0 )		sub[ INDEX( xx+i, yy+j, zz, lenx, leny ) ] = in[ INDEX(x+i, y+j, z, ldx, ldy) ];	
		if( j != 0 && k != 0 )		sub[ INDEX( xx, yy+j, zz+k, lenx, leny ) ] = in[ INDEX(x, y+j, z+k, ldx, ldy) ];
		if( i != 0 && k != 0 )		sub[ INDEX( xx+i, yy, zz+k, lenx, leny ) ] = in[ INDEX(x+i, y, z+k, ldx, ldy) ];
		if( i != 0 && j != 0 && k != 0 )	sub[ INDEX( xx+i, yy+j, zz+k, lenx, leny ) ] = in[ INDEX(x+i, y+j, z+k, ldx, ldy) ];
		
		__syncthreads(); 
		
        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * sub[INDEX(xx, yy, zz, lenx, leny)] \
            + ALPHA_NZZ * sub[INDEX(xx-1, yy, zz, lenx, leny)] \
            + ALPHA_PZZ * sub[INDEX(xx+1, yy, zz, lenx, leny)] \
            + ALPHA_ZNZ * sub[INDEX(xx, yy-1, zz, lenx, leny)] \
            + ALPHA_ZPZ * sub[INDEX(xx, yy+1, zz, lenx, leny)] \
            + ALPHA_ZZN * sub[INDEX(xx, yy, zz-1, lenx, leny)] \
            + ALPHA_ZZP * sub[INDEX(xx, yy, zz+1, lenx, leny)] \
            + ALPHA_NNZ * sub[INDEX(xx-1, yy-1, zz, lenx, leny)] \
            + ALPHA_PNZ * sub[INDEX(xx+1, yy-1, zz, lenx, leny)] \
            + ALPHA_NPZ * sub[INDEX(xx-1, yy+1, zz, lenx, leny)] \
            + ALPHA_PPZ * sub[INDEX(xx+1, yy+1, zz, lenx, leny)] \
            + ALPHA_NZN * sub[INDEX(xx-1, yy, zz-1, lenx, leny)] \
            + ALPHA_PZN * sub[INDEX(xx+1, yy, zz-1, lenx, leny)] \
            + ALPHA_NZP * sub[INDEX(xx-1, yy, zz+1, lenx, leny)] \
            + ALPHA_PZP * sub[INDEX(xx+1, yy, zz+1, lenx, leny)] \
            + ALPHA_ZNN * sub[INDEX(xx, yy-1, zz-1, lenx, leny)] \
            + ALPHA_ZPN * sub[INDEX(xx, yy+1, zz-1, lenx, leny)] \
            + ALPHA_ZNP * sub[INDEX(xx, yy-1, zz+1, lenx, leny)] \
            + ALPHA_ZPP * sub[INDEX(xx, yy+1, zz+1, lenx, leny)] \
            + ALPHA_NNN * sub[INDEX(xx-1, yy-1, zz-1, lenx, leny)] \
            + ALPHA_PNN * sub[INDEX(xx+1, yy-1, zz-1, lenx, leny)] \
            + ALPHA_NPN * sub[INDEX(xx-1, yy+1, zz-1, lenx, leny)] \
            + ALPHA_PPN * sub[INDEX(xx+1, yy+1, zz-1, lenx, leny)] \
            + ALPHA_NNP * sub[INDEX(xx-1, yy-1, zz+1, lenx, leny)] \
            + ALPHA_PNP * sub[INDEX(xx+1, yy-1, zz+1, lenx, leny)] \
            + ALPHA_NPP * sub[INDEX(xx-1, yy+1, zz+1, lenx, leny)] \
            + ALPHA_PPP * sub[INDEX(xx+1, yy+1, zz+1, lenx, leny)];
    }
}


ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size (ceiling(nx, XX), ceiling(ny, YY), ceiling(nz, ZZ));
    dim3 block_size (XX, YY, ZZ);
    for(int t = 0; t < nt; ++t) {
        stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}