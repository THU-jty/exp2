#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include "common.h"

const char* version_name = "Optimized version";

/* your implementation */
void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
	if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
	
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
	
	for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
		if( rank == pnum-1 ){
			MPI_Isend( a0+ldy*ldx, ldy*ldx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[0] );
			MPI_Irecv( a0, ldx*ldy, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &req[1] );
			MPI_Waitall( 2, req, sta );
		}
		else if( rank == 0 ){
			MPI_Isend( a0+ldy*ldx*lz, ldy*ldx, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &req[2] );
			MPI_Irecv( a0+ldy*ldx*(lz+1), ldx*ldy, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[3] );
			MPI_Waitall( 2, &req[2], &sta[2] );
		}
		else{
			MPI_Isend( a0+ldy*ldx, ldy*ldx, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[0] );
			MPI_Irecv( a0, ldx*ldy, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &req[1] );
			
			MPI_Isend( a0+ldy*ldx*lz, ldy*ldx, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &req[2] );
			MPI_Irecv( a0+ldy*ldx*(lz+1), ldx*ldy, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[3] );
			MPI_Waitall( 4, req, sta );
		}
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
						+ ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];
				}
			}
        }
	}
	return buffer[nt % 2];
}

/* your implementation */
ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    return grid;
}