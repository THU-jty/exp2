#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char **argv)
{
	MPI_Comm active_procs;
	int pid, pnum;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
	printf("this is process in %d of %d\n", pid, pnum);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Request req[2];
	MPI_Status sta[2];
	int tmp;
	MPI_Isend( &pid, 1, MPI_INT, (pid+1)%pnum, 0, MPI_COMM_WORLD, &req[0] );
	MPI_Irecv( &tmp, 1, MPI_INT, (pid+pnum-1)%pnum, 0, MPI_COMM_WORLD, &req[1] );
	MPI_Waitall( 2, req, sta );
	printf("%d recv mesg from %d\n", pid, tmp);
}