#include "common.h"
#include <omp.h>
#define XX 64
#define YY 64
#define ZZ 16
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

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
    grid_info->halo_size_x = \
    grid_info->halo_size_y = \
    grid_info->halo_size_z = 2;
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
	int t_sp = cir, y_sp, z_sp = lz, x_sp = 128;
	if( ly == 256 ) y_sp = 16;
	if( ly == 384 ) y_sp = 12;
	if( ly == 512 ) y_sp = 8;

	int llx = x_sp+2*cir;
	double *a;
	double* p[30][20][3];
	int numy = ly/y_sp;
	int num_t = 24;
	a = ( double * )malloc( sizeof(double)*(t_sp-1)*llx*(y_sp+2*cir)*3*num_t );
	for( int k = 0; k < num_t; k ++ )
        for( int i = 0; i < t_sp-1; i ++ ){
            for( int j = 0; j < 3; j ++ ){
                p[k][i+1][j] = &a[ (i*3+j)*llx*(y_sp+2*cir)+k*(t_sp-1)*llx*(y_sp+2*cir)*3 ];
            }
        }
	int nw = 0;
	omp_set_num_threads(num_t);
	for( int tt = 0; tt < nt; tt += t_sp ){	
		ptr_t a0 = buffer[nw];
        ptr_t a1 = buffer[nw^1];
		nw ^= 1;
		//int step = min( t_sp, nt-tt );
		int step = t_sp;
		for( int zz = z_start; zz < z_end; zz += z_sp ){
			#pragma omp parallel for collapse(2) schedule (dynamic)
			for( int yy = y_start; yy < y_end; yy += y_sp ){//进行y z两层block
				for( int xx = x_start; xx < x_end; xx += x_sp ){
					int id = omp_get_thread_num();
					for( int w = 1; w < t_sp; w ++ )
						for( int kk = 0; kk < 3; kk ++ )
							for( int j = 0; j < y_sp+cir*2; j ++ )
								for( int i = 0; i < llx; i ++ )
									p[id][w][kk][INDEX(i, j, 0, llx, 0)] = 0.0;
					for( int k = zz-cir; k < zz+z_sp+cir; k ++ ){
						int ss = min( step, (k-zz+cir)/2 );//ss 为要迭代的步数
						p[id][0][k%3] = &a0[INDEX(xx-cir, yy-cir, k, ldx, ldy)];
	#ifdef __OUT
//                        for( int j = 0; j < y_sp+2*cir; j ++ ){
//                            for( int i = 0; i < llx; i ++ ){
//                                printf("%f ", p[id][0][k%3])
//                            }
//                        }
	#endif

						if( ss <= 0 ) continue;
						if( ss == step )
							p[id][ss][(k-ss)%3] = &a1[INDEX(xx-cir, yy-cir, k-ss, ldx, ldy)];
						for( int s = 1; s <= ss; s ++ ){
                            int px, prex;
							if( s == step ) px = ldx;
							else px = llx;
							if( s-1 == 0 ) prex = ldx;
							else prex = llx;
							if( k-s < z_start || k-s >= z_end ){
								for( int y = 0; y < y_sp+2*cir; y ++ )
									for( int x = 0; x < x_sp+2*cir; x ++ )
										p[id][s][(k-s)%3][INDEX(x, y, 0, px, ldy)] = 0.0;
								continue;
							}
							int y0, y1, x0, x1;
							int s0, s1, s2;
							s0 = (k-s)%3;
							s1 = (k-s-1)%3;
							s2 = (k-s+1)%3;
	#ifdef __OUT
							printf("x %d y %d z %d step %d:\n", xx, yy, k-s, s);
	#endif
							y0 = max(yy-cir+s,y_start)-yy+cir;
							y1 = min(yy+y_sp+cir-s, y_end)+cir-yy;
							x0 = max(xx-cir+s,x_start)-xx+cir;
							x1 = min(xx+x_sp+cir-s, x_end)+cir-xx;
							for( int y = y0; y < y1; y ++ ){
								//for( int x = s; x < lx+2*cir-s; x ++ ){p[id][s-1][ ( k-s )%3 ][x+prex*y]
								for( int x = x0; x < x1; x ++ ){
									p[id][s][s0][INDEX(x, y, 0, px, ldy)] \
									= ALPHA_ZZZ * p[id][s-1][s0][INDEX(x, y, 0, prex, ldy)]
									+ ALPHA_NZZ * p[id][s-1][s0][INDEX(x-1, y, 0, prex, ldy)]
									+ ALPHA_PZZ * p[id][s-1][s0][INDEX(x+1, y, 0, prex, ldy)]
									+ ALPHA_ZNZ * p[id][s-1][s0][INDEX(x, y-1, 0, prex, ldy)]
									+ ALPHA_ZPZ * p[id][s-1][s0][INDEX(x, y+1, 0, prex, ldy)]
									+ ALPHA_ZZN * p[id][s-1][s1][INDEX(x, y, 0, prex, ldy)]
									+ ALPHA_ZZP * p[id][s-1][s2][INDEX(x, y, 0, prex, ldy)];
								}
							}
	#ifdef __OUT
                            for( int y = 0; y < y_sp+2*cir; y ++ ){
                                for( int x = 0; x < x_sp+2*cir; x ++ ){
                                    int px;
                                    if( s == 0 || s == step ) px = ldx;
                                    else px = llx;
                                    printf("%10f ", p[id][s][(k-s)%3][INDEX(x, y, 0, px, ldy)]);
                                }
                                puts("");
                            }
                            puts("");
                            printf("pre down\n");
                            for( int y = 0; y < y_sp+2*cir; y ++ ){
                                for( int x = 0; x < x_sp+2*cir; x ++ ){
                                    int px;
                                    if( s-1 == 0 || s-1 == step ) px = ldx;
                                    else px = llx;
                                    printf("%10f ", p[id][s-1][(k-s)%3][INDEX(x, y, 0, px, ldy)]);
                                }
                                puts("");
                            }
                            puts("");
	#endif
						}
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
	int lx = grid_info->local_size_x;
	int ly = grid_info->local_size_y;
	int lz = grid_info->local_size_z;

	int cir = grid_info->halo_size_x;
	int t_sp = cir, y_sp = 8, z_sp = lz, x_sp = 64;

	int llx = x_sp+2*cir;
	double *a;
	double* p[30][20][3];
	int numy = ly/y_sp;
	int num_t = 24;
	a = ( double * )malloc( sizeof(double)*(t_sp-1)*llx*(y_sp+2*cir)*3*num_t );
	for( int k = 0; k < num_t; k ++ )
        for( int i = 0; i < t_sp-1; i ++ ){
            for( int j = 0; j < 3; j ++ ){
                p[k][i+1][j] = &a[ (i*3+j)*llx*(y_sp+2*cir)+k*(t_sp-1)*llx*(y_sp+2*cir)*3 ];
            }
        }
	int nw = 0;
	omp_set_num_threads(num_t);
	for( int tt = 0; tt < nt; tt += t_sp ){	
		ptr_t a0 = buffer[nw];
        ptr_t a1 = buffer[nw^1];
		nw ^= 1;
		//int step = min( t_sp, nt-tt );
		int step = t_sp;
		for( int zz = z_start; zz < z_end; zz += z_sp ){
			#pragma omp parallel for collapse(2) schedule (dynamic)
			for( int yy = y_start; yy < y_end; yy += y_sp ){//进行y z两层block
				for( int xx = x_start; xx < x_end; xx += x_sp ){
					int id = omp_get_thread_num();
					for( int w = 1; w < t_sp; w ++ )
						for( int kk = 0; kk < 3; kk ++ )
							for( int j = 0; j < y_sp+cir*2; j ++ )
								for( int i = 0; i < llx; i ++ )
									p[id][w][kk][INDEX(i, j, 0, llx, 0)] = 0.0;
					for( int k = zz-cir; k < zz+z_sp+cir; k ++ ){
						int ss = min( step, (k-zz+cir)/2 );//ss 为要迭代的步数
						p[id][0][k%3] = &a0[INDEX(xx-cir, yy-cir, k, ldx, ldy)];
	#ifdef __OUT
//                        for( int j = 0; j < y_sp+2*cir; j ++ ){
//                            for( int i = 0; i < llx; i ++ ){
//                                printf("%f ", p[id][0][k%3])
//                            }
//                        }
	#endif

						if( ss <= 0 ) continue;
						if( ss == step )
							p[id][ss][(k-ss)%3] = &a1[INDEX(xx-cir, yy-cir, k-ss, ldx, ldy)];
						for( int s = 1; s <= ss; s ++ ){
                            int px, prex;
							if( s == step ) px = ldx;
							else px = llx;
							if( s-1 == 0 ) prex = ldx;
							else prex = llx;
							if( k-s < z_start || k-s >= z_end ){
								for( int y = 0; y < y_sp+2*cir; y ++ )
									for( int x = 0; x < x_sp+2*cir; x ++ )
										p[id][s][(k-s)%3][INDEX(x, y, 0, px, ldy)] = 0.0;
								continue;
							}
							int y0, y1, x0, x1;
							int s0, s1, s2;
							s0 = (k-s)%3;
							s1 = (k-s-1)%3;
							s2 = (k-s+1)%3;
	#ifdef __OUT
							printf("x %d y %d z %d step %d:\n", xx, yy, k-s, s);
	#endif
							y0 = max(yy-cir+s,y_start)-yy+cir;
							y1 = min(yy+y_sp+cir-s, y_end)+cir-yy;
							x0 = max(xx-cir+s,x_start)-xx+cir;
							x1 = min(xx+x_sp+cir-s, x_end)+cir-xx;
							for( int y = y0; y < y1; y ++ ){
								//for( int x = s; x < lx+2*cir-s; x ++ ){p[id][s-1][ ( k-s )%3 ][x+prex*y]
								for( int x = x0; x < x1; x ++ ){
									p[id][s][(k-s)%3][INDEX(x, y, 0, px, ldy)] 
									= ALPHA_ZZZ * p[id][s-1][ ( k-s )%3 ][INDEX(x, y, 0, prex, ldy)] \
									+ ALPHA_NZZ * p[id][s-1][ ( k-s )%3 ][INDEX(x-1, y, 0, prex, ldy)] \
									+ ALPHA_PZZ * p[id][s-1][ ( k-s )%3 ][INDEX(x+1, y, 0, prex, ldy)] \
									+ ALPHA_ZNZ * p[id][s-1][ ( k-s )%3 ][INDEX(x, y-1, 0, prex, ldy)] \
									+ ALPHA_ZPZ * p[id][s-1][ ( k-s )%3 ][INDEX(x, y+1, 0, prex, ldy)] \
									+ ALPHA_ZZN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x, y, 0, prex, ldy)] \
									+ ALPHA_ZZP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x, y, 0, prex, ldy)] \
									+ ALPHA_NNZ * p[id][s-1][ ( k-s )%3 ][INDEX(x-1, y-1, 0, prex, ldy)] \
									+ ALPHA_PNZ * p[id][s-1][ ( k-s )%3 ][INDEX(x+1, y-1, 0, prex, ldy)] \
									+ ALPHA_NPZ * p[id][s-1][ ( k-s )%3 ][INDEX(x-1, y+1, 0, prex, ldy)] \
									+ ALPHA_PPZ * p[id][s-1][ ( k-s )%3 ][INDEX(x+1, y+1, 0, prex, ldy)] \
									+ ALPHA_NZN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x-1, y, 0, prex, ldy)] \
									+ ALPHA_PZN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x+1, y, 0, prex, ldy)] \
									+ ALPHA_NZP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x-1, y, 0, prex, ldy)] \
									+ ALPHA_PZP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x+1, y, 0, prex, ldy)] \
									+ ALPHA_ZNN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x, y-1, 0, prex, ldy)] \
									+ ALPHA_ZPN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x, y+1, 0, prex, ldy)] \
									+ ALPHA_ZNP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x, y-1, 0, prex, ldy)] \
									+ ALPHA_ZPP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x, y+1, 0, prex, ldy)] \
									+ ALPHA_NNN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x-1, y-1, 0, prex, ldy)] \
									+ ALPHA_PNN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x+1, y-1, 0, prex, ldy)] \
									+ ALPHA_NPN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x-1, y+1, 0, prex, ldy)] \
									+ ALPHA_PPN * p[id][s-1][ ( k-s-1 )%3 ][INDEX(x+1, y+1, 0, prex, ldy)] \
									+ ALPHA_NNP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x-1, y-1, 0, prex, ldy)] \
									+ ALPHA_PNP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x+1, y-1, 0, prex, ldy)] \
									+ ALPHA_NPP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x-1, y+1, 0, prex, ldy)] \
									+ ALPHA_PPP * p[id][s-1][ ( k-s+1 )%3 ][INDEX(x+1, y+1, 0, prex, ldy)];								

								}
							}
	#ifdef __OUT
                            for( int y = 0; y < y_sp+2*cir; y ++ ){
                                for( int x = 0; x < x_sp+2*cir; x ++ ){
                                    int px;
                                    if( s == 0 || s == step ) px = ldx;
                                    else px = llx;
                                    printf("%10f ", p[id][s][(k-s)%3][INDEX(x, y, 0, px, ldy)]);
                                }
                                puts("");
                            }
                            puts("");
                            printf("pre down\n");
                            for( int y = 0; y < y_sp+2*cir; y ++ ){
                                for( int x = 0; x < x_sp+2*cir; x ++ ){
                                    int px;
                                    if( s-1 == 0 || s-1 == step ) px = ldx;
                                    else px = llx;
                                    printf("%10f ", p[id][s-1][(k-s)%3][INDEX(x, y, 0, px, ldy)]);
                                }
                                puts("");
                            }
                            puts("");
	#endif
						}
					}
				}

			}
		}
	}

    return buffer[nw];
}
