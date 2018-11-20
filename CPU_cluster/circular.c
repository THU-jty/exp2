#include "common.h"
#include <omp.h>
#define XX 64
#define YY 64
#define ZZ 16
#define min(a,b) (((a)<(b))?(a):(b))

const char* version_name = "A naive base-line";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* Naive implementation uses Process 0 to do all computations */
    if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 4;
    grid_info->halo_size_y = 4;
    grid_info->halo_size_z = 4;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;
	int lx = grid_info->local_size_x;
	int ly = grid_info->local_size_y;
	int lz = grid_info->local_size_z;
	
	int cir = grid_info->halo_size_x;
	int t_sp = cir, y_sp = 4, z_sp = lz;
	double *a;
	double* p[10][3];
	a = ( double * )malloc( sizeof(double)*(t_sp-1)*ldx*(y_sp+2*cir)*3 );
	for( int i = 0; i < t_sp-1; i ++ ){
		for( int j = 0; j < 3; j ++ ){
			p[i+1][j] = &a[ (i*3+j)*ldx*(y_sp+2*cir) ];
		}
	}
	int nw = 0;
	for( int tt = 0; tt < nt; tt += t_sp ){
		ptr_t a0 = buffer[nw];
        ptr_t a1 = buffer[nw^1];
		nw ^= 1;
		int step = min( t_sp, nt-tt );
		for( int zz = z_start; zz < z_end; zz += z_sp ){
			for( int yy = y_start; yy < y_end; yy += y_sp ){//进行y z两层block
				
				for( int k = zz-cir; k < zz+z_sp+cir; k ++ ){
					int ss = min( step, (k-zz)/2 );//ss 为要迭代的步数
					p[0][k%3] = &a0[INDEX(0, yy-cir, k, ldx, ldy)];
					if( ss <= 0 ) continue;
					if( ss == step )
						p[ss][(k-ss)%3] = &a1[INDEX(0, yy-cir, k-ss, ldx, ldy)];
					for( int s = 1; s <= ss; s ++ ){
						for( int y = s; y < y_sp+2*cir-s; y ++ )
							for( int x = s; x < lx+2*cir-s; x ++ )
								p[s][(k-s)%3][INDEX(x, y, 0, ldx, ldy)] \
								= ALPHA_ZZZ * p[s-1][ ( k-s )%3 ][INDEX(x, y, 0, ldx, ldy)] \
								+ ALPHA_NZZ * p[s-1][ ( k-s )%3 ][INDEX(x-1, y, 0, ldx, ldy)] \
								+ ALPHA_PZZ * p[s-1][ ( k-s )%3 ][INDEX(x+1, y, 0, ldx, ldy)] \
								+ ALPHA_ZNZ * p[s-1][ ( k-s )%3 ][INDEX(x, y-1, 0, ldx, ldy)] \
								+ ALPHA_ZPZ * p[s-1][ ( k-s )%3 ][INDEX(x, y+1, 0, ldx, ldy)] \
								+ ALPHA_ZZN * p[s-1][ ( k-(s+1) )%3 ][INDEX(x, y, 0, ldx, ldy)] \
								+ ALPHA_ZZP * p[s-1][ ( k-(s-1) )%3 ][INDEX(x, y, 0, ldx, ldy)];
					}
				}
				
			}
		}
	}
		
    return buffer[nw];
}

/*
#pragma omp parallel for schedule (dynamic)
        for(int y = y_start; y < y_end; ++y) {
			for(int z = z_start; z < z_end; ++z) {
				for(int x = x_start; x < x_end; ++x) {
					a1[INDEX(x, y, z, ldx, ldy)] \
						= ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
						+ ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
						+ ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
						+ ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
						+ ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
						+ ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
						+ ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];
				}
			}
        }
*/

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;
    
	omp_set_num_threads(24);
	for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
#pragma omp parallel for schedule (static)		
        for(int z = z_start; z < z_end; ++z) {
            for(int y = y_start; y < y_end; ++y) {
                for(int x = x_start; x < x_end; ++x) {
                    a1[INDEX(x, y, z, ldx, ldy)] \
                        = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
                        + ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
                        + ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
                        + ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
                        + ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
                        + ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
                        + ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)] \
                        + ALPHA_NNZ * a0[INDEX(x-1, y-1, z, ldx, ldy)] \
                        + ALPHA_PNZ * a0[INDEX(x+1, y-1, z, ldx, ldy)] \
                        + ALPHA_NPZ * a0[INDEX(x-1, y+1, z, ldx, ldy)] \
                        + ALPHA_PPZ * a0[INDEX(x+1, y+1, z, ldx, ldy)] \
                        + ALPHA_NZN * a0[INDEX(x-1, y, z-1, ldx, ldy)] \
                        + ALPHA_PZN * a0[INDEX(x+1, y, z-1, ldx, ldy)] \
                        + ALPHA_NZP * a0[INDEX(x-1, y, z+1, ldx, ldy)] \
                        + ALPHA_PZP * a0[INDEX(x+1, y, z+1, ldx, ldy)] \
                        + ALPHA_ZNN * a0[INDEX(x, y-1, z-1, ldx, ldy)] \
                        + ALPHA_ZPN * a0[INDEX(x, y+1, z-1, ldx, ldy)] \
                        + ALPHA_ZNP * a0[INDEX(x, y-1, z+1, ldx, ldy)] \
                        + ALPHA_ZPP * a0[INDEX(x, y+1, z+1, ldx, ldy)] \
                        + ALPHA_NNN * a0[INDEX(x-1, y-1, z-1, ldx, ldy)] \
                        + ALPHA_PNN * a0[INDEX(x+1, y-1, z-1, ldx, ldy)] \
                        + ALPHA_NPN * a0[INDEX(x-1, y+1, z-1, ldx, ldy)] \
                        + ALPHA_PPN * a0[INDEX(x+1, y+1, z-1, ldx, ldy)] \
                        + ALPHA_NNP * a0[INDEX(x-1, y-1, z+1, ldx, ldy)] \
                        + ALPHA_PNP * a0[INDEX(x+1, y-1, z+1, ldx, ldy)] \
                        + ALPHA_NPP * a0[INDEX(x-1, y+1, z+1, ldx, ldy)] \
                        + ALPHA_PPP * a0[INDEX(x+1, y+1, z+1, ldx, ldy)];
                }
            }
        }
    }
    return buffer[nt % 2];
}