# !/bin/bash
export DAPL_DBG_TYPE=0
DATAPATH=/home/course/HW2/data
a=( 1,2,4,8,16,24 )
for i in ${a[@]};do
	export OMP_NUM_THREADS=i
	srun -N 1 ./benchmark-naive		 7 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256 | grep -e Per -e stencil | tee 1
	srun -N 1 ./benchmark-timeskew 	 7 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256 | grep -e Per -e stencil | tee 1
	srun -N 1 ./benchmark-cir        7 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256 | grep -e Per -e stencil | tee 1
done

for i in ${a[@]};
do
	export OMP_NUM_THREADS=i
	srun -N 1 ./benchmark-naive		 7 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384 | grep -e Per -e stencil | tee 1
	srun -N 1 ./benchmark-timeskew 	 7 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384 | grep -e Per -e stencil | tee 1
	srun -N 1 ./benchmark-cir        7 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384 | grep -e Per -e stencil | tee 1
done

for i in ${a[@]};
do
	export OMP_NUM_THREADS=i
	srun -N 1 ./benchmark-naive		 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512 | grep -e Per -e stencil | tee 1
	srun -N 1 ./benchmark-timeskew 	 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512 | grep -e Per -e stencil | tee 1
	srun -N 1 ./benchmark-cir        7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512 | grep -e Per -e stencil | tee 1
done