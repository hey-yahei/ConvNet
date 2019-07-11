mymodel:
	g++ tests/test_mymodel.cpp \
	src/Concat.cpp src/Convolution.cpp src/Eltwise.cpp src/FullyConnection.cpp src/Pooling.cpp \
	src/helper.cpp src/TarFile.cpp \
	-I ./include -o build/mymodel