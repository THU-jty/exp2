#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include "common.h"

const char* version_name = "mpi";

/* your implementation */
void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
	int pnum = grid_info->p_num;
	grid_info->local_size_x = grid_info->global_size_x;
	grid_info->local_size_y = grid_info->global_size_y;
	if( grid_info->p_id != grid_info->p_num-1 ){
        grid_info->local_size_z = grid_info->global_size_z/grid_info->p_num;
	}
	else{
        grid_info->local_size_z = grid_info->global_size_z-grid_info->global_size_z/pnum*(pnum-1);
	}
	
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = grid_info->p_id*(grid_info->global_size_z/pnum);
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

/* your implementation */
void destroy_dist_grid(dist_grid_info_t *grid_info) {

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
	int rank = grid_info->p_id, pnum = grid_info->p_num;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//printf("rank %d pnum %d\n", rank, pnum);
	MPI_Status  sta[4];
    MPI_Request req[4];
	MPI_Datatype plane;
    MPI_Type_vector(ly,lx,ldx,MPI_DOUBLE,&plane);
    MPI_Type_commit(&plane);	
		
	int XX = lx, ZZ = lz;
	int YY = 16; 
	if( lx == 384 ) YY = 12;
	if( lx == 512 ) YY = 8;
	
	//omp_set_num_threads(24);
	for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
		//if( pnum == 1 ){  }
		 if( rank == pnum-1 ){
			MPI_Isend( a0+ldy*ldx+ldx+1, 1, plane, rank-1, 0, MPI_COMM_WORLD, &req[0] );
			MPI_Irecv( a0+ldx+1, 1, plane, rank-1, 1, MPI_COMM_WORLD, &req[1] );
			MPI_Wait( &req[0], &sta[0] );
			MPI_Wait( &req[1], &sta[1] );
		}
		else if( rank == 0 ){
			MPI_Isend( a0+ldy*ldx*lz+ldx+1, 1, plane, rank+1, 1, MPI_COMM_WORLD, &req[2] );
			MPI_Irecv( a0+ldy*ldx*(lz+1)+ldx+1, 1, plane, rank+1, 0, MPI_COMM_WORLD, &req[3] );
			MPI_Wait( &req[2], &sta[2] );
			MPI_Wait( &req[3], &sta[3] );
		}
		else{	
			
			MPI_Isend( a0+ldy*ldx+ldx+1, 1, plane, rank-1, 0, MPI_COMM_WORLD, &req[0] );
			MPI_Irecv( a0+ldx+1, 1, plane, rank-1, 1, MPI_COMM_WORLD, &req[1] );
			MPI_Wait( &req[0], &sta[0] );
			MPI_Wait( &req[1], &sta[1] );
			
			MPI_Isend( a0+ldy*ldx*lz+ldx+1, 1, plane, rank+1, 1, MPI_COMM_WORLD, &req[2] );
			MPI_Irecv( a0+ldy*ldx*(lz+1)+ldx+1, 1, plane, rank+1, 0, MPI_COMM_WORLD, &req[3] );
			MPI_Wait( &req[2], &sta[2] );
			MPI_Wait( &req[3], &sta[3] );
		}
//#pragma omp parallel for schedule (dynamic)
        for(int zz = z_start; zz < z_end; zz += ZZ) {
			for( int yy = y_start; yy < y_end; yy += YY ){
					#pragma omp parallel for schedule (dynamic)
					for(int z = zz; z < zz+ZZ; ++z)
					for(int y = yy; y < yy+YY; ++y) {
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
	int rank = grid_info->p_id, pnum = grid_info->p_num;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//printf("rank %d pnum %d\n", rank, pnum);
	MPI_Status  sta[4];
    MPI_Request req[4];
	MPI_Datatype plane;
    MPI_Type_vector(ly,lx,ldx,MPI_DOUBLE,&plane);
    MPI_Type_commit(&plane);
	
	int XX = lx, ZZ = lz;
	int YY = 4096/XX; 
	if( lx == 384 ) YY = 12;
	if( lx == 192 ) YY = 24;
	//omp_set_num_threads(24);
	for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
		//if( pnum == 1 ){  }
		 if( rank == pnum-1 ){
			MPI_Isend( a0+ldy*ldx+ldx+1, 1, plane, rank-1, 0, MPI_COMM_WORLD, &req[0] );
			MPI_Irecv( a0+ldx+1, 1, plane, rank-1, 1, MPI_COMM_WORLD, &req[1] );
			MPI_Waitall( 2, req, sta );
		}
		else if( rank == 0 ){
			MPI_Isend( a0+ldy*ldx*lz+ldx+1, 1, plane, rank+1, 1, MPI_COMM_WORLD, &req[2] );
			MPI_Irecv( a0+ldy*ldx*(lz+1)+ldx+1, 1, plane, rank+1, 0, MPI_COMM_WORLD, &req[3] );
			MPI_Waitall( 2, &req[2], &sta[2] );
		}
		else{
			MPI_Isend( a0+ldy*ldx+ldx+1, 1, plane, rank-1, 0, MPI_COMM_WORLD, &req[0] );
			MPI_Irecv( a0+ldx+1, 1, plane, rank-1, 1, MPI_COMM_WORLD, &req[1] );
			
			MPI_Isend( a0+ldy*ldx*lz+ldx+1, 1, plane, rank+1, 1, MPI_COMM_WORLD, &req[2] );
			MPI_Irecv( a0+ldy*ldx*(lz+1)+ldx+1, 1, plane, rank+1, 0, MPI_COMM_WORLD, &req[3] );
			MPI_Waitall( 4, req, sta );
		}

        for(int zz = z_start; zz < z_end; zz += ZZ) {
			for( int yy = y_start; yy < y_end; yy += YY ){
				for( int xx = x_start; xx < x_end; xx += XX ){
					#pragma omp parallel for schedule (dynamic)
					for(int z = zz; z < zz+ZZ; ++z)
					for(int y = yy; y < yy+YY; ++y) {
						for(int x = xx; x < xx+XX; ++x) {
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
        }
	}
	return buffer[nt % 2];
}