#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include "common.h"

const char* version_name = "Optimized version";

/* your implementation */
void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
	int pnum = grid_info->p_num, id = grid_info->p_id;
	if( pnum != 8 && pnum != 64 ){ printf("argv error\n"); exit(0); }
	int n;
	if( pnum == 8 ) n = 2;
	else n = 4;
	int xx = id%n, yy = (id/n)%n, zz = (id/n/n)%n;
	grid_info->local_size_x = grid_info->global_size_x/n;
	grid_info->local_size_y = grid_info->global_size_y/n;
	grid_info->local_size_z = grid_info->global_size_z/n;
	
    grid_info->offset_x = xx*grid_info->local_size_x;
    grid_info->offset_y = yy*grid_info->local_size_y;
    grid_info->offset_z = zz*grid_info->local_size_z;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

/* your implementation */
void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

inline void cal( cptr_t a0, ptr_t a1, int x, int y, int z, int ldx, int ldy )
{
	a1[INDEX(x, y, z, ldx, ldy)] \
	= ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
	+ ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
	+ ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
	+ ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
	+ ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
	+ ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
	+ ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];
}

inline void cal27( cptr_t a0, ptr_t a1, int x, int y, int z, int ldx, int ldy )
{
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

/* your implementation */
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
	int rank = grid_info->p_id, pnum = grid_info->p_num, n = ((pnum==64)?4:2);
	int rk_z0 = rank-n*n, rk_z1 = rank+n*n, rk_y0 = rank-n, rk_y1 = rank+n,
	rk_x0 = rank-1, rk_x1 = rank+1;
	int rkx = rank%n, rky = (rank/n)%n, rkz = (rank/n/n)%n;
	int px = n, py = n, pz = n;
	
	//printf("rank %d pnum %d\n", rank, pnum);
	MPI_Status  sta[16];
    MPI_Request req[16];
	MPI_Datatype xyplane, yzplane, xzplane;
    MPI_Type_vector(1,ldx*ldy,0,MPI_DOUBLE,&xyplane);
    MPI_Type_commit(&xyplane);
	
	MPI_Type_vector(ldz,ldx,ldx*ldy,MPI_DOUBLE,&xzplane);
    MPI_Type_commit(&xzplane);
	
	MPI_Type_vector(ldy*ldz,1,ldx,MPI_DOUBLE,&yzplane);
    MPI_Type_commit(&yzplane);
	
	omp_set_num_threads(24);
	for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
#pragma omp parallel sections
		{		
			#pragma omp section
			{
				if( rkz != 0 && pz != 1 ){
					MPI_Isend( a0+ldy*ldx, 1, xyplane, rk_z0, 0, MPI_COMM_WORLD, &req[0] );
					MPI_Irecv( a0, 1, xyplane, rk_z0, 1, MPI_COMM_WORLD, &req[1] );
				}
				if( rkz != pz-1 && pz != 1 ){
					MPI_Isend( a0+ldy*ldx*lz, 1, xyplane, rk_z1, 1, MPI_COMM_WORLD, &req[2] );
					MPI_Irecv( a0+ldy*ldx*(lz+1), 1, xyplane, rk_z1, 0, MPI_COMM_WORLD, &req[3] );
				}
				if( rky != 0 && py != 1 ){
					MPI_Isend( a0+ldx, 1, xzplane, rk_y0, 2, MPI_COMM_WORLD, &req[4] );
					MPI_Irecv( a0, 1, xzplane, rk_y0, 3, MPI_COMM_WORLD, &req[5] );
				}
				if( rky != py-1 && py != 1 ){
					MPI_Isend( a0+ly*ldx, 1, xzplane, rk_y1, 3, MPI_COMM_WORLD, &req[6] );
					MPI_Irecv( a0+(ly+1)*ldx, 1, xzplane, rk_y1, 2, MPI_COMM_WORLD, &req[7] );
				}
				if( rkx != 0 && px != 1 ){
					MPI_Isend( a0+1, 1, yzplane, rk_x0, 5, MPI_COMM_WORLD, &req[8] );
					MPI_Irecv( a0, 1, yzplane, rk_x0, 6, MPI_COMM_WORLD, &req[9] );
				}
				if( rkx != px-1 && px != 1 ){
					MPI_Isend( a0+lx, 1, yzplane, rk_x1, 6, MPI_COMM_WORLD, &req[10] );
					MPI_Irecv( a0+lx+1, 1, yzplane, rk_x1, 5, MPI_COMM_WORLD, &req[11] );
				}
				
				if( rkz != 0 && pz != 1 ){
					MPI_Waitall( 2, &req[0], &sta[0] );
				}
				if( rkz != pz-1 && pz != 1 ){
					MPI_Waitall( 2, &req[2], &sta[2] );
				}
				if( rky != 0 && py != 1 ){
					MPI_Waitall( 2, &req[4], &sta[4] );
				}
				if( rky != py-1 && py != 1 ){
					MPI_Waitall( 2, &req[6], &sta[6] );
				}
				if( rkx != 0 && px != 1 ){
					MPI_Waitall( 2, &req[8], &sta[8] );
				}
				if( rkx != px-1 && px != 1 ){
					MPI_Waitall( 2, &req[10], &sta[10] );
				}
						//cal z
				for( int z = z_start; z < z_start+1; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal( a0, a1, x, y, z, ldx, ldy );	
				for( int z = z_end-1; z < z_end; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal( a0, a1, x, y, z, ldx, ldy );		

				//cal y
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_start; y < y_start+1; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal( a0, a1, x, y, z, ldx, ldy );		
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_end-1; y < y_end; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal( a0, a1, x, y, z, ldx, ldy );		
				
				//cal x
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_start; x < x_start+1; ++x) 
							cal( a0, a1, x, y, z, ldx, ldy );		
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_end-1; x < x_end; ++x) 
							cal( a0, a1, x, y, z, ldx, ldy );	
			}
			#pragma omp section
			{
				for(int z = z_start+1; z < z_end-1; ++z) 
					for(int y = y_start+1; y < y_end-1; ++y) 
						for(int x = x_start+1; x < x_end-1; ++x) 
							cal( a0, a1, x, y, z, ldx, ldy );
			}
		}
	}
	return buffer[nt % 2];
}


/* your implementation */
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
	int rank = grid_info->p_id, pnum = grid_info->p_num, n = ((pnum==64)?4:2);
	int rk_z0 = rank-n*n, rk_z1 = rank+n*n, rk_y0 = rank-n, rk_y1 = rank+n,
	rk_x0 = rank-1, rk_x1 = rank+1;
	int rkx = rank%n, rky = (rank/n)%n, rkz = (rank/n/n)%n;
	int px = n, py = n, pz = n;
	
	//printf("rank %d pnum %d\n", rank, pnum);
	MPI_Status  sta[16];
    MPI_Request req[16];
	MPI_Datatype xyplane, yzplane, xzplane;
    MPI_Type_vector(1,ldx*ldy,0,MPI_DOUBLE,&xyplane);
    MPI_Type_commit(&xyplane);
	
	MPI_Type_vector(ldz,ldx,ldx*ldy,MPI_DOUBLE,&xzplane);
    MPI_Type_commit(&xzplane);
	
	MPI_Type_vector(ldy*ldz,1,ldx,MPI_DOUBLE,&yzplane);
    MPI_Type_commit(&yzplane);
	
	omp_set_num_threads(24);
	for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
#pragma omp parallel sections
		{		
			#pragma omp section
			{
				if( rkz != 0 && pz != 1 ){
					MPI_Isend( a0+ldy*ldx, 1, xyplane, rk_z0, 0, MPI_COMM_WORLD, &req[0] );
					MPI_Irecv( a0, 1, xyplane, rk_z0, 1, MPI_COMM_WORLD, &req[1] );
				}
				if( rkz != pz-1 && pz != 1 ){
					MPI_Isend( a0+ldy*ldx*lz, 1, xyplane, rk_z1, 1, MPI_COMM_WORLD, &req[2] );
					MPI_Irecv( a0+ldy*ldx*(lz+1), 1, xyplane, rk_z1, 0, MPI_COMM_WORLD, &req[3] );
				}
				
				if( rkz != 0 && pz != 1 ){
					MPI_Waitall( 2, &req[0], &sta[0] );
				}
				if( rkz != pz-1 && pz != 1 ){
					MPI_Waitall( 2, &req[2], &sta[2] );
				}
				
				if( rky != 0 && py != 1 ){
					MPI_Isend( a0+ldx, 1, xzplane, rk_y0, 2, MPI_COMM_WORLD, &req[4] );
					MPI_Irecv( a0, 1, xzplane, rk_y0, 3, MPI_COMM_WORLD, &req[5] );
				}
				if( rky != py-1 && py != 1 ){
					MPI_Isend( a0+ly*ldx, 1, xzplane, rk_y1, 3, MPI_COMM_WORLD, &req[6] );
					MPI_Irecv( a0+(ly+1)*ldx, 1, xzplane, rk_y1, 2, MPI_COMM_WORLD, &req[7] );
				}
				
				if( rky != 0 && py != 1 ){
					MPI_Waitall( 2, &req[4], &sta[4] );
				}
				if( rky != py-1 && py != 1 ){
					MPI_Waitall( 2, &req[6], &sta[6] );
				}
				
				if( rkx != 0 && px != 1 ){
					MPI_Isend( a0+1, 1, yzplane, rk_x0, 5, MPI_COMM_WORLD, &req[8] );
					MPI_Irecv( a0, 1, yzplane, rk_x0, 6, MPI_COMM_WORLD, &req[9] );
				}
				if( rkx != px-1 && px != 1 ){
					MPI_Isend( a0+lx, 1, yzplane, rk_x1, 6, MPI_COMM_WORLD, &req[10] );
					MPI_Irecv( a0+lx+1, 1, yzplane, rk_x1, 5, MPI_COMM_WORLD, &req[11] );
				}
				
				if( rkx != 0 && px != 1 ){
					MPI_Waitall( 2, &req[8], &sta[8] );
				}
				if( rkx != px-1 && px != 1 ){
					MPI_Waitall( 2, &req[10], &sta[10] );
				}
						//cal z
				for( int z = z_start; z < z_start+1; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal27( a0, a1, x, y, z, ldx, ldy );	
				for( int z = z_end-1; z < z_end; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal27( a0, a1, x, y, z, ldx, ldy );		

				//cal y
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_start; y < y_start+1; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal27( a0, a1, x, y, z, ldx, ldy );		
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_end-1; y < y_end; ++y) 
						for(int x = x_start; x < x_end; ++x) 
							cal27( a0, a1, x, y, z, ldx, ldy );		
				
				//cal x
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_start; x < x_start+1; ++x) 
							cal27( a0, a1, x, y, z, ldx, ldy );		
				for( int z = z_start; z < z_end; ++z )
					for(int y = y_start; y < y_end; ++y) 
						for(int x = x_end-1; x < x_end; ++x) 
							cal27( a0, a1, x, y, z, ldx, ldy );	
			}
			#pragma omp section
			{
				for(int z = z_start+1; z < z_end-1; ++z) 
					for(int y = y_start+1; y < y_end-1; ++y) 
						for(int x = x_start+1; x < x_end-1; ++x) 
							cal27( a0, a1, x, y, z, ldx, ldy );
			}
		}
	}
	return buffer[nt % 2];
}