#include "helper.h"
#include "TarFile.h"
#include <iostream>
#include <fstream>
// #include <io.h>   // VS
#include <vector>

using namespace std;

Tree* DecodeHuffman(Bit& accessor, TarFile* tarfile) {
	Tree* left, * right;
	unsigned char value = 0;

	switch (accessor.Next()) {
	case -1:
		return nullptr;
	case 0:
		left = DecodeHuffman(accessor, tarfile);
		right = DecodeHuffman(accessor, tarfile);
		return new Tree(-1, left, right);
	case 1:
		value = 0;
		for (int i = 0; i < 8; i++) {
			value <<= 1;
			value += accessor.Next();
		}
		return new Tree(value, nullptr, nullptr);
	default:
		cerr << "DecodeHuffman: unknown bit" << endl;
		exit(1);
	}
}

Tree* DecodeHuffmanCodebook(const char* path, TarFile* tarfile) {
	Bit accessor(path, tarfile);
	return DecodeHuffman(accessor, tarfile);
}

unsigned long LoadFromFile(const char* path, char* dst, TarFile* tarfile) {
	unsigned long byte_cnt = 0;
	if (nullptr != tarfile) {
		tarfile->GetFileContents(path, dst);
		byte_cnt = tarfile->GetFileSize(path);
	} else {
		char temp;
		ifstream in_file(path, ios::in | ios::binary);

		if (!in_file) {
			cerr << "Open file failed: " << path << endl;
			return 0;
		}
		while (in_file.read((char*)& temp, sizeof(temp))) {
			dst[byte_cnt] = temp;
			byte_cnt++;
		}
	}

	return byte_cnt;
}


unsigned long GetFileLength(const char* path) {
	FILE* fp;
	fp =fopen(path, "r");
	if (nullptr == fp) {
		cerr << "GetFileLength: open file fail" << endl;
		exit(1);
	}
	fseek(fp, 0L, SEEK_END);
	long size = ftell(fp);
	fclose(fp);
	return size;
}


unsigned long LoadFromHuffmanFile(string prefix, vector<unsigned char> &dst, TarFile* tarfile) {
	string data = prefix + ".hf.dat";
	string codebook = prefix + ".hfcb.dat";
	int cnt = 0;

	Tree* huffman_tree = DecodeHuffmanCodebook(codebook.c_str(), tarfile);
	Tree* ptr = huffman_tree;
	Bit accessor(data.c_str(), tarfile);
	int bit;
	while ((bit = accessor.Next()) != -1) {
		if (0 == bit) {
			ptr = ptr->left;
		} else if (1 == bit) {
			ptr = ptr->right;
		} else {
			cerr << "LoadFromHuffmanFile: unknown bit" << endl;
			exit(1);
		}

		if (-1 != ptr->value) {
			dst.push_back(ptr->value);
			cnt++;
			ptr = huffman_tree;
		}
	}

	delete huffman_tree;
	return cnt;
}

unsigned long LoadFromSparseHuffmanFile(string prefix, char* dst, TarFile* tarfile) {
	string data_path = prefix + ".data";
	string index_path = prefix + ".index";

	vector<unsigned char> index, data;
	LoadFromHuffmanFile(index_path, index, tarfile);
	LoadFromHuffmanFile(data_path, data, tarfile);
	unsigned long size = index.size();
	if (size != data.size()) {
		cerr << "LoadFromSparseHuffmanFile: size not match" << endl;
		exit(1);
	}

	index[0]--;
	for (int i = 0; i < size; i++) {
		dst += index[i];
		*dst = data[i];
	}
	return size;
}

unsigned long SaveToFile(const char* path, char* src, unsigned long bytes) {
	ofstream out_file(path, ios::out | ios::binary);
	if (!out_file) {
		cerr << "Open result file failed." << endl;
		return 0;
	}
	out_file.write(src, bytes);
	return bytes;
}
