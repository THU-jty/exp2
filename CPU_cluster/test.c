#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


int main(int argc, char **argv)
{
omp_set_num_threads(24);
omp_set_nested(1);
#pragma omp parallel sections
{
	#pragma omp section
	{
		#pragma omp parallel for
		for( int i = 0; i < 24; i ++ ){
			printf("1 %d %d\n", i, omp_get_thread_num());
		}
	}
	#pragma omp section
	{
		#pragma omp parallel for
		for( int i = 0; i < 24; i ++ ){
			printf("2 %d %d\n", i, omp_get_thread_num());
		}
	}
}
#pragma omp parallel for
		for( int i = 0; i < 24; i ++ ){
			printf("3 %d %d\n", i, omp_get_thread_num());
		}

}