cuda_files := tcgpudb.cu, tcgpusa.cu, test.cu
obj_files := tcgpudb tcgpusa test

tcgpudb:
	rm -f tcgpudb
	nvcc -Xcompiler -fopenmp tcgpudb.cu -O3 -o tcgpudb
	./tcgpudb

tcgpusa:
	rm -f tcgpusa
	nvcc -Xcompiler -fopenmp tcgpusa.cu -O3 -o tcgpusa 
	./tcgpusa

test: 
	rm -f test
	nvcc -Xcompiler -fopenmp test.cu -O3 -o test 
	./test

test2: 
	rm -f test2
	nvcc -Xcompiler -fopenmp test2.cu -O3 -o test2
	./test2

db:
	rm -f db
	nvcc -Xcompiler -fopenmp db.cu -O3 -o db 
	./db

sa:
	rm -f sa
	nvcc -Xcompiler -fopenmp sa.cu -O3 -o sa
	./sa


graph:
	rm -f graph
	nvcc -Xcompiler -fopenmp graph.cu -O3 -o graph
	./graph

clear:
	rm -f $(obj_files)

.PHONY: tcgpudb tcgpusa test db graph test2 sa