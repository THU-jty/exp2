# !/bin/bash
export DAPL_DBG_TYPE=0
DATAPATH=/home/course/HW2/data
a=(1 2 3 4 5 6 7 8)
fil=3

for i in ${a[@]};do
	export OMP_NUM_THREADS=24
	srun -N $i ./benchmark-mpiz		 7 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256 | grep -e Per -e stencil | tee -a $fil
	srun -N $i ./benchmark-mpiz		 7 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384 | grep -e Per -e stencil | tee -a $fil
	srun -N $i ./benchmark-mpiz		 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512 | grep -e Per -e stencil | tee -a $fil
	srun -N $i ./benchmark-mpiz		 27 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256 | grep -e Per -e stencil | tee -a $fil
	srun -N $i ./benchmark-mpiz		 27 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384 | grep -e Per -e stencil | tee -a $fil
	srun -N $i ./benchmark-mpiz		 27 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512 | grep -e Per -e stencil | tee -a $fil	
done
