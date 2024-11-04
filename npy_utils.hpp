#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace npy {

    struct NpyArray {
        NpyArray(const std::vector<size_t>& _shape, size_t _word_size, const bool _fortran_order) :
            shape(_shape), word_size(_word_size), fortran_order(_fortran_order)
        {
            num_vals = 1;
            for (const unsigned long i: shape)
                num_vals *= i;
            data_holder = std::make_shared<std::vector<char>>(num_vals * word_size);
        }

        NpyArray() : shape(0), word_size(0), fortran_order(false), num_vals(0) {}

        template<typename T>
        T* data()
        {
            return reinterpret_cast<T*>(&(*data_holder)[0]);
        }

        template<typename T>
        const T* data() const
        {
            return reinterpret_cast<T*>(&(*data_holder)[0]);
        }

        template<typename T>
        std::vector<T> as_vec() const
        {
            const T* p = data<T>();
            return std::vector<T>(p, p + num_vals);
        }

        [[nodiscard]] size_t num_bytes() const { return data_holder->size(); }

        std::shared_ptr<std::vector<char>> data_holder;
        std::vector<size_t> shape;
        size_t word_size;
        bool fortran_order;
        size_t num_vals;
    };

    using npz_t = std::map<std::string, NpyArray>;

    void parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);
    auto npy_load(const std::string& fname) -> NpyArray;
    auto load_npy_arr(const std::string& fname) -> std::tuple<std::unique_ptr<char[]>, size_t, size_t>;

    template<typename T>
    auto load_npy_mat(const std::string& npy_file)
    {
        npy::NpyArray npy_data = npy::npy_load(npy_file);
        std::vector<size_t> shape = npy_data.shape;

        if (shape.size() != 2) {
            throw std::runtime_error("Only 2D arrays can be converted to Eigen matrices.");
        }

        T* raw_data = npy_data.data<T>();

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(shape[0], shape[1]);

        if (npy_data.fortran_order) {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    matrix(i, j) = raw_data[j * shape[0] + i];
                }
            }
        } else {
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    matrix(i, j) = raw_data[i * shape[1] + j];
                }
            }
        }
        return matrix;
    }

    template<typename T, int fortran_order>
    void save_mat(const std::string& filename, const Eigen::Matrix<T, -1, -1, fortran_order>& matrix)
    {
        std::cout << "START" << std::endl;
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        // Write the NPY file magic string
        constexpr char magic_string[] = "\x93NUMPY";
        outfile.write(magic_string, 6);

        // Write version number (major and minor)
        constexpr uint8_t major_version = 2;
        constexpr uint8_t minor_version = 0;
        outfile.write(reinterpret_cast<const char*>(&major_version), 1);
        outfile.write(reinterpret_cast<const char*>(&minor_version), 1);

        // Build the header string
        std::string header = "{'descr': '";

        // Choose the dtype depending on the type T
        if (std::is_same<T, float>::value) {
            header += "<f4"; // little-endian 32-bit float
        } else if (std::is_same<T, double>::value) {
            header += "<f8"; // little-endian 64-bit float
        } else if (std::is_same<T, int8_t>::value) {
            header += "|i1"; // char
        } else if (std::is_same<T, int16_t>::value) {
            header += "<i2"; // little-endian 32-bit integer
        } else if (std::is_same<T, int32_t>::value) {
            header += "<i4"; // little-endian 32-bit integer
        } else if (std::is_same<T, int64_t>::value) {
            header += "<i8"; // little-endian 64-bit integer
        } else if (std::is_same<T, uint8_t>::value) {
            header += "|u1"; // unsigned 8-bit integer (no endianness marker for 1-byte types)
        } else if (std::is_same<T, uint16_t>::value) {
            header += "<u2"; // little-endian 16-bit unsigned integer
        } else if (std::is_same<T, uint32_t>::value) {
            header += "<u4"; // little-endian 32-bit unsigned integer
        } else if (std::is_same<T, uint64_t>::value) {
            header += "<u8"; // little-endian 64-bit unsigned integer
        } else {
            std::cerr << "Unsupported data type" << std::endl;
            outfile.close();
            return;
        }

        // Check if the matrix is Fortran-ordered (column-major)
        bool is_fortran_order = !matrix.IsRowMajor;

        header += "', 'fortran_order': ";
        header += (is_fortran_order ? "True" : "False");
        header += ", 'shape': (" + std::to_string(matrix.rows()) + ", " + std::to_string(matrix.cols()) + "), }";

        // Align header to a 16-byte boundary
        const size_t padding = 16 - (10 + header.size()) % 16;
        header.append(padding, ' ');
        header += '\n'; // Must end with newline

        const auto header_len = static_cast<uint32_t>(header.size());

        // Write the header length
        outfile.write(reinterpret_cast<const char*>(&header_len), 4);

        // Write the header
        outfile.write(header.c_str(), static_cast<long>(header.size()));

        // Write the matrix data
        outfile.write(reinterpret_cast<const char*>(matrix.data()), static_cast<long>(matrix.size() * sizeof(T)));

        // Close the file
        outfile.close();
        std::cout << "Saved matrix to: " << filename << std::endl;
    }

    template<typename T>
    void save_arr(const std::string& filename, const T* data, std::size_t size_v)
    {
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        // Write the NPY file magic string
        constexpr char magic_string[] = "\x93NUMPY";
        outfile.write(magic_string, 6);

        // Write version number (major and minor)
        constexpr uint8_t major_version = 2;
        constexpr uint8_t minor_version = 0;
        outfile.write(reinterpret_cast<const char*>(&major_version), 1);
        outfile.write(reinterpret_cast<const char*>(&minor_version), 1);

        // Build the header string
        std::string header = "{'descr': '";

        if (std::is_same<T, float>::value) {
            header += "<f4"; // little-endian 32-bit float
        } else if (std::is_same<T, double>::value) {
            header += "<f8"; // little-endian 64-bit float
        } else if (std::is_same<T, int8_t>::value) {
            header += "|i1"; // char
        } else if (std::is_same<T, int16_t>::value) {
            header += "<i2"; // little-endian 32-bit integer
        } else if (std::is_same<T, int32_t>::value) {
            header += "<i4"; // little-endian 32-bit integer
        } else if (std::is_same<T, int64_t>::value) {
            header += "<i8"; // little-endian 64-bit integer
        } else if (std::is_same<T, uint8_t>::value) {
            header += "|u1"; // unsigned 8-bit integer (no endianness marker for 1-byte types)
        } else if (std::is_same<T, uint16_t>::value) {
            header += "<u2"; // little-endian 16-bit unsigned integer
        } else if (std::is_same<T, uint32_t>::value) {
            header += "<u4"; // little-endian 32-bit unsigned integer
        } else if (std::is_same<T, uint64_t>::value) {
            header += "<u8"; // little-endian 64-bit unsigned integer
        } else {
            std::cerr << "Unsupported data type" << std::endl;
            outfile.close();
            return;
        }

        header += "', 'fortran_order': False, 'shape': (" + std::to_string(size_v) + ",), }";

        // Align header to a 16-byte boundary
        const size_t padding = 16 - (10 + header.size()) % 16;
        header.append(padding, ' ');
        header += '\n'; // Must end with newline

        const auto header_len = static_cast<uint32_t>(header.size());

        // Write the header length
        outfile.write(reinterpret_cast<const char*>(&header_len), 4);

        // Write the header
        outfile.write(header.c_str(), static_cast<long>(header.size()));

        // Write the vector data (raw pointer data)
        outfile.write(reinterpret_cast<const char*>(data), static_cast<long>(size_v * sizeof(T)));

        // Close the file
        outfile.close();
    }

    template<typename T>
    void save_arr_as_matrix(const std::string& filename, const T* const data, std::size_t size_h, std::size_t size_w)
    {
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }

        // Write the NPY file magic string
        constexpr char magic_string[] = "\x93NUMPY";
        outfile.write(magic_string, 6);

        // Write version number (major and minor)
        constexpr uint8_t major_version = 2;
        constexpr uint8_t minor_version = 0;
        outfile.write(reinterpret_cast<const char*>(&major_version), 1);
        outfile.write(reinterpret_cast<const char*>(&minor_version), 1);

        // Build the header string
        std::string header = "{'descr': '";

        // Choose the dtype depending on the type T
        if (std::is_same<T, float>::value) {
            header += "<f4"; // little-endian 32-bit float
        } else if (std::is_same<T, double>::value) {
            header += "<f8"; // little-endian 64-bit float
        } else if (std::is_same<T, int8_t>::value) {
            header += "|i1"; // char
        } else if (std::is_same<T, int16_t>::value) {
            header += "<i2"; // little-endian 32-bit integer
        } else if (std::is_same<T, int32_t>::value) {
            header += "<i4"; // little-endian 32-bit integer
        } else if (std::is_same<T, int64_t>::value) {
            header += "<i8"; // little-endian 64-bit integer
        } else if (std::is_same<T, uint8_t>::value) {
            header += "|u1"; // unsigned 8-bit integer (no endianness marker for 1-byte types)
        } else if (std::is_same<T, uint16_t>::value) {
            header += "<u2"; // little-endian 16-bit unsigned integer
        } else if (std::is_same<T, uint32_t>::value) {
            header += "<u4"; // little-endian 32-bit unsigned integer
        } else if (std::is_same<T, uint64_t>::value) {
            header += "<u8"; // little-endian 64-bit unsigned integer
        } else {
            std::cerr << "Unsupported data type" << std::endl;
            outfile.close();
            return;
        }

        header += "', 'fortran_order': False";
        header += ", 'shape': (" + std::to_string(size_h) + ", " + std::to_string(size_w) + "), }";

        // Align header to a 16-byte boundary
        const size_t padding = 16 - (10 + header.size()) % 16;
        header.append(padding, ' ');
        header += '\n'; // Must end with newline

        const auto header_len = static_cast<uint32_t>(header.size());

        // Write the header length
        outfile.write(reinterpret_cast<const char*>(&header_len), 4);

        // Write the header
        outfile.write(header.c_str(), static_cast<long>(header.size()));

        // Write the vector data (raw pointer data)
        outfile.write(reinterpret_cast<const char*>(data), static_cast<long>(size_h * size_w * sizeof(T)));
        std::cout << "Saved matrix to: " << filename << std::endl;
        // Close the file
        outfile.close();

    }

    inline void _map_data(const std::string& fname, void* dst)
    {
        FILE* fp = fopen(fname.c_str(), "rb");
        if (!fp)
            throw std::runtime_error("npy_load_data: Unable to open file " + fname);
        std::vector<size_t> shape;
        size_t word_size;
        bool fortran_order;
        npy::parse_npy_header(fp, word_size, shape, fortran_order);
        size_t total_size = 1;
        for (size_t s: shape)
            total_size *= s;
        size_t n_bytes = word_size * total_size;
        const size_t nread = fread(dst, 1, n_bytes, fp);
        if (nread != n_bytes)
            throw std::runtime_error("npy_load_data: failed fread");
        fclose(fp);
    }

    // Main function to read .npy files and stack them into an Eigen matrix
    template<typename T, int FORTRAN_ORDER>
    auto npy_folder2mat(const std::string& folder_name, const std::string& prefix, const int start_i, const std::string& suffix)
            -> Eigen::Matrix<T, -1, -1, FORTRAN_ORDER>
    {
        // Initialize variables
        std::vector<size_t> shape;
        size_t word_size;
        bool fortran_order;

        // Read the first file to get matrix dimensions and order
        std::string first_file = folder_name + "/" + prefix + std::to_string(start_i) + suffix;
        FILE* fp = fopen(first_file.c_str(), "rb");
        if (!fp)
            throw std::runtime_error("Unable to open file " + first_file);
        npy::parse_npy_header(fp, word_size, shape, fortran_order);
        fclose(fp);

        // Check if the fortran order matches the template parameter
        if (!fortran_order != FORTRAN_ORDER)
            throw std::runtime_error(
                    "npy_folder2mat: Matrix order mismatch. Expected fortran order does not match file order.");

        // Determine matrix dimensions
        size_t rows = shape[0];
        size_t cols = shape[1];

        // Calculate the total rows for the Eigen matrix by counting files
        size_t file_count = 0;
        int _i = start_i;
        while (true)
        {
            std::string file_name = folder_name + "/";
            file_name += prefix + std::to_string(_i);
            file_name += suffix;
            FILE* test_fp = fopen(file_name.c_str(), "rb");
            if (!test_fp)
                break;
            fclose(test_fp);
            file_count++;
            _i++;
        }

        // Create the Eigen matrix with the appropriate size
        Eigen::Matrix<T, -1, -1, FORTRAN_ORDER> eigen_matrix(rows * file_count, cols);
        T* data_ptr = eigen_matrix.data();

        // Load each file into the matrix
        for (size_t i = start_i; i < file_count; ++i) {
            std::string file_name = folder_name + "/";
            file_name += prefix + std::to_string(i);
            file_name += suffix;
            _map_data(file_name, data_ptr + i * rows * cols);
        }

        return eigen_matrix;
    }

} // namespace npy

#endif
