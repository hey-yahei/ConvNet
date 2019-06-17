1. 编译libzteai.so
    `make zte_ai`

2. 编译zte_output
    `make zte_output`

3. 运行测试模型，对test_2000_images_list.txt的第<start>至<end-1>行图片，输出文件至outputs目录
    `export LD_LIBRARY_PATH=./build`
    `./build/zte_output <start> <end>`
            —— 如`./build/zte_output 0 100`测试输出前一百行（两百张图片）的结果

4. 计算输出文件的TPR@0.0001
    `python3 tests/get_tpr.py`