#if 0

#include "TarFile.h"

int test_tar()
{
	const std::string tar_file_path{ "../models/test.tar" };
	TarFile tarfile(tar_file_path.c_str());

	bool is_valid_tar_file = tarfile.IsValidTarFile();
	if (!is_valid_tar_file) {
		fprintf(stderr, "it is not a valid tar file: %s\n", tar_file_path.c_str());
		return -1;
	}

	fprintf(stderr, "tar file size: %d byte\n", tarfile.GetTarSize());

	std::vector<std::string> file_names = tarfile.GetFileNames();
	fprintf(stderr, "tar file count: %d\n", file_names.size());
	for (auto name : file_names) {
		fprintf(stderr, "=====================================\n");
		size_t file_size = tarfile.GetFileSize(name.c_str());
		fprintf(stderr, "file name: %s,  size: %d byte\n", name.c_str(), file_size);

		char* contents = new char[file_size + 1];
		tarfile.GetFileContents(name.c_str(), contents);
		contents[file_size] = '\0';

		fprintf(stderr, "contents:\n%s\n", contents);
		delete[] contents;
	}
	return 0;
}

int main() {
	return test_tar();
}

#endif