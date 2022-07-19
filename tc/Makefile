cuda_files := tcgpudb.cu, tcgpusa.cu, test.cu
obj_files := tcgpudb test test2 sa1 sa2 sa3 hybrid

tcgpudb:
	rm -f tcgpudb
	nvcc -Xcompiler -fopenmp -lineinfo tcgpudb.cu -O3 -o tcgpudb
	./tcgpudb

test: 
	rm -f test
	nvcc -Xcompiler -fopenmp -lineinfo test.cu -O3 -o test 
	./test

test2: 
	rm -f test2
	nvcc -Xcompiler -fopenmp -lineinfo test2.cu -O3 -o test2
	./test2

sa1:
	rm -f sa1
	nvcc -Xcompiler -fopenmp -lineinfo sa1.cu -O3 -o sa1
	./sa1

sa2:
	rm -f sa2
	nvcc -Xcompiler -fopenmp -lineinfo sa2.cu -O3 -o sa2
	./sa2

sa3:
	rm -f sa3
	nvcc -Xcompiler -fopenmp -lineinfo sa3.cu -O3 -o sa3
	./sa3

hybrid:
	rm -f hybrid
	nvcc -Xcompiler -fopenmp -lineinfo hybrid.cu -O3 -o hybrid
	./hybrid

tc:
	rm -f triangleCount
	nvcc -Xcompiler -fopenmp -lineinfo triangleCount.cu -O3 -o triangleCount
	./triangleCount

clear:
	rm -f $(obj_files)

.PHONY: tcgpudb tcgpusa test db graph test2 sa1 sa2 sa3 hybrid tc