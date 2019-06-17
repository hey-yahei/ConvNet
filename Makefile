zte_ai: 
	g++ tests/ZTE_AI.cpp src/CBConvolution.cpp src/Eltwise.cpp src/CBFullyConnection.cpp \
	src/Convolution.cpp src/helper.cpp src/TarFile.cpp -I ./include -o build/libzteai.so -shared -fPIC -O2

zte_test:
	g++ tests/zte_test.cpp -L ./build -lzteai `pkg-config --cflags opencv` `pkg-config --libs opencv` \
	-I ./include -o build/zte_test

zte_output:
	g++ tests/zte_output.cpp -L ./build -lzteai `pkg-config --cflags opencv` `pkg-config --libs opencv` \
	-I ./include -o build/zte_output

zte_test_from_file:
	g++ tests/zte_test_from_file.cpp -L ./build -lzteai `pkg-config --cflags opencv` `pkg-config --libs opencv` \
	-I ./include -o build/zte_test_from_file