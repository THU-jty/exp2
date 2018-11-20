#define ALPHA_ZZZ (0.9415)
#define ALPHA_NZZ (0.01531)
#define ALPHA_PZZ (0.02345)
#define ALPHA_ZNZ (-0.01334)
#define ALPHA_ZPZ (-0.03512)
#define ALPHA_ZZN (0.02333)
#define ALPHA_ZZP (0.02111)
#define ALPHA_NNZ (-0.03154)
#define ALPHA_PNZ (-0.01234)
#define ALPHA_NPZ (0.01111)
#define ALPHA_PPZ (0.02222)
#define ALPHA_NZN (0.01212)
#define ALPHA_PZN (0.01313)
#define ALPHA_NZP (-0.01242)
#define ALPHA_PZP (-0.03751)
#define ALPHA_ZNN (-0.03548)
#define ALPHA_ZPN (-0.04214)
#define ALPHA_ZNP (0.01795)
#define ALPHA_ZPP (0.01279)
#define ALPHA_NNN (0.01537)
#define ALPHA_PNN (-0.01357)
#define ALPHA_NPN (-0.01734)
#define ALPHA_PPN (0.01975)
#define ALPHA_NNP (0.02568)
#define ALPHA_PNP (0.02734)
#define ALPHA_NPP (-0.01242)
#define ALPHA_PPP (-0.02018)

#define INDEX(xx, yy, zz, ldxx, ldyy) ((xx) + (ldxx) * ((yy) + (ldyy) * (zz)))

typedef double data_t;
typedef data_t* ptr_t;
typedef const data_t* cptr_t;
#define DATA_TYPE MPI_DOUBLE

/* 
 * Global array `g`: array of size
 * global_size_x * global_size_y * global_size_z
 *
 * Local array `l`: array of size
 * (local_size_x + halo_size_x * 2) * (local_size_y + halo_size_y * 2) * (local_size_y + halo_size_x * 2)
 *
 * the element `l[halo_size_x + x][halo_size_y + y][halo_size_z + z]` in local array represents 
 * the element `g[offset_x + x][offset_y + y][offset_z + z]` in global array 
 */
typedef struct {
    int global_size_x, global_size_y, global_size_z;
    int local_size_x, local_size_y, local_size_z;
    int offset_x, offset_y, offset_z;
    int halo_size_x, halo_size_y, halo_size_z;
    int p_id, p_num;
    void *additional_info;
} dist_grid_info_t;

/* type == 7 or type == 27 */
void create_dist_grid(dist_grid_info_t *info, int stencil_type);
void destroy_dist_grid(dist_grid_info_t *info);
/* `arr` is the input array, `aux` is an auxiliary buffer
 * return the pointer to the output array
 * the returned value should be either equal to `arr` or `aux` */
ptr_t stencil_7(ptr_t arr, ptr_t aux, const dist_grid_info_t *info, int nt);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

//extern const char* version_name;


typedef struct {
    data_t norm_1, norm_2, norm_inf;
} check_result_t;

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
	int t_sp = cir, y_sp = 4, z_sp = 64;
	double *a;
	double* p[10][3];
	a = ( double * )malloc( sizeof(double)*(t_sp-1)*ldx*(y_sp+2*cir)*3 );
	for( int i = 0; i < t_sp-2; i ++ ){
		for( int j = 0; j < 3; j ++ ){
			p[i+1][j] = &a[ (i*3+j)*ldx*(y_sp+2*cir) ];
		}
	}
	int nw = 0;
	for( int tt = 0; tt < nt; tt += t_sp ){
		cptr_t a0 = buffer[nw];
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
						p[ss][(k-ss%3+3)%3] = &a1[INDEX(0, yy-cir, k-ss, ldx, ldy)];
					for( int s = 1; s <= ss; s ++ ){
						for( int y = -cir+s; y < y_sp+cir-s; y ++ )
							for( int x = -cir+s; x < lx+cir-s; x ++ )
								p[s][(k-s%3+3)%3][INDEX(x, y, 0, ldx, ldy)] \
								= ALPHA_ZZZ * p[s-1][ ( k-s%3+3 )%3 ][INDEX(x, y, 0, ldx, ldy)] \
								+ ALPHA_NZZ * p[s-1][ ( k-s%3+3 )%3 ][INDEX(x-1, y, 0, ldx, ldy)] \
								+ ALPHA_PZZ * p[s-1][ ( k-s%3+3 )%3 ][INDEX(x+1, y, 0, ldx, ldy)] \
								+ ALPHA_ZNZ * p[s-1][ ( k-s%3+3 )%3 ][INDEX(x, y-1, 0, ldx, ldy)] \
								+ ALPHA_ZPZ * p[s-1][ ( k-s%3+3 )%3 ][INDEX(x, y+1, 0, ldx, ldy)] \
								+ ALPHA_ZZN * p[s-1][ ( k-(s+1)%3+3 )%3 ][INDEX(x, y, 0, ldx, ldy)] \
								+ ALPHA_ZZP * p[s-1][ ( k-(s-1)%3+3 )%3 ][INDEX(x, y, 0, ldx, ldy)];
					}
				}
				
			}
		}
	}
		
    return buffer[nw];
}

int main(int argc, char **argv) {
    int nt, type, thread_scheme, status, p_id;
    dist_grid_info_t info;
    dist_grid_info_t *grid_info;
    double start, end, pre_time;
    ptr_t a0, a1, ans0, ans1;
	
	
	info.global_size_x = \
	info.global_size_y = \
	info.global_size_z = 256;
	
	create_dist_grid( &info, 0 );
	int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;
	
	a0 = ( double* )malloc( sizeof(double)*ldx*ldy*ldz );
	a1 = ( double* )malloc( sizeof(double)*ldx*ldy*ldz );
	stencil_7( a0, a1, &info, 16 )
	
	
    return 0;
}





