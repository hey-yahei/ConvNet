#ifndef _HELPER_H
#define _HELPER_H

#include <string>
#include "TarFile.h"

using namespace std;

unsigned long GetFileLength(const char* path);
unsigned long LoadFromFile(const char* path, char* dst, TarFile* tarfile = nullptr);

class Bit {
private:
	unsigned char* buffer = nullptr;
	unsigned char* ptr;
	int size;
	int idx;
	int num_pad;
public:
	Bit(){}
	Bit(const char* path, TarFile* tarfile = nullptr) {
		size = (nullptr != tarfile) ? tarfile->GetFileSize(path) : GetFileLength(path);
		buffer = new unsigned char[size];
		if (nullptr != tarfile)
			tarfile->GetFileContents(path, (char*)buffer);
		else
			LoadFromFile(path, (char*)buffer);
		num_pad = buffer[0];
		ptr = buffer + 1;
		idx = 0;
	}
	int Next() {
		if ( (ptr >= buffer + size) || (ptr == buffer + size - 1) && (idx + num_pad >= 8) )
			return -1;

		int ret = (*ptr & (1 << (7 - idx++))) ? 1 : 0;
		if (idx >= 8) {
			idx = 0;
			ptr++;
		}

		return ret;
	}
	~Bit(){ delete buffer; }
};

class Tree {
public:
	Tree* left = nullptr;
	Tree* right = nullptr;
	int value;
	
	Tree(){}
	Tree(int v, Tree* l, Tree* r): value(v), left(l), right(r){}
	~Tree() {
		delete left;
		delete right;
	}
};

unsigned long SaveToFile(const char* path, char* src, unsigned long bytes);
Tree* DecodeHuffman(Bit& accessor, TarFile* tarfile = nullptr);
Tree* DecodeHuffmanCodebook(const char* path, TarFile* tarfile = nullptr);
unsigned long LoadFromHuffmanFile(string prefix, vector<unsigned char>& dst, TarFile* tarfile = nullptr);
unsigned long LoadFromSparseHuffmanFile(string prefix, char* dst, TarFile* tarfile = nullptr);

#endif	// ~~~_HELPER_H