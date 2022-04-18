EXECUTABLES = vectorVectorProduct matrixVectorProduct jacobi2d

all: $(EXECUTABLES)

vectorVectorProduct: vectorVectorProduct.cu
	nvcc -std=c++11  vectorVectorProduct.cu -o vectorVectorProduct -Xcompiler -fopenmp

matrixVectorProduct: matrixVectorProduct.cu
	nvcc -std=c++11  matrixVectorProduct.cu -o matrixVectorProduct -Xcompiler -fopenmp

jacobi2d: jacobi2d.cu
	nvcc -std=c++11  jacobi2d.cu -o jacobi2d -Xcompiler -fopenmp

clean:
	rm -rf $(EXECUTABLES)