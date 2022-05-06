cuda_files := tcgpudb.cu, tcgpusa.cu, test.cu
obj_files := tcgpudb tcgpusa test

tcgpudb:
	rm -f tcgpudb
	nvcc -Xcompiler -fopenmp tcgpudb.cu -O3 -o tcgpudb

tcgpusa:
	rm -f tcgpusa
	nvcc -Xcompiler -fopenmp tcgpusa.cu -O3 -o tcgpusa 

test: 
	rm -f test
	nvcc -Xcompiler -fopenmp test.cu -O3 -o test 

run_test:
	./test

run_db:
	./tcgpudb

run_sa: 
	./tcgpusa

clear:
	rm -f $(obj_files)

.PHONY: tcgpudb tcgpusa test