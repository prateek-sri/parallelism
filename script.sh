module load cuda
export OMP_NUM_THREADS=64
#./parboil compile sgemm base -v     
#./parboil compile sgemm omp_base -v
#./parboil compile sgemm cuda -v
#./parboil compile sgemm cuda_base -v
#./parboil compile spmv cuda_base_tex -v
#./parboil compile spmv cuda_tex -v
#./parboil compile spmv cpu -v
#./parboil compile spmv omp_base -v
#./parboil compile spmv cuda_base -v
#./parboil compile spmv cuda -v
#./parboil compile sad base -v
#./parboil compile sad cpu -v
#./parboil compile sad cuda_base -v
inputs=(small medium)
for i in "${inputs[@]}"
do
    ./parboil run sgemm base $i -v     
    ./parboil run sgemm cuda $i -v
    ./parboil run sgemm cuda_base $i -v
#    ./parboil run sgemm omp_base $i -v
done

inputs=(small medium large)
for i in "${inputs[@]}"
do
    ./parboil run spmv cuda_base_tex $i -v
    ./parboil run spmv cuda_tex $i -v
    ./parboil run spmv cuda_base $i -v
    ./parboil run spmv cuda $i -v
    ./parboil run spmv cpu $i -v
#    ./parboil run spmv omp_base $i -v
done 

inputs=(default large)
for i in "${inputs[@]}"
do
    ./parboil run sad base $i -v
    ./parboil run sad cuda_base $i -v
    ./parboil run sad cpu $i -v
done


