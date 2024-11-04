#include "npy_utils.hpp"


auto find_header_substring(const char* str) -> std::string
{
    using namespace std;
    const char* start = nullptr;
    const char* end = nullptr;

    if (!str) {
        return "";
    }

    // Find the first non-null character to start the substring
    for (const char* ptr = str; *ptr != '\n'; ++ptr) {
        if (*ptr != '\0') {
            start = ptr;
            break;
        }
    }

    if (!start) {
        return "";
    }

    for (const char* ptr = start; *ptr != '\n'; ++ptr) {
        end = ptr;
    }

    // If we found both a start and end, return the substring
    if (end) {
        string header(start, end - start + 2);
        return header;
    }

    // If no newline was found, return an empty string
    return "";
}

void npy::parse_npy_header(FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order)
{
    char buffer[256];
    const size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11)
        throw std::runtime_error("parse_npy_header: failed in fread");

    auto _ = fgets(buffer, 256, fp);

    auto header = find_header_substring(buffer);

    if (header[header.size() - 1] != '\n') {
        // print header[header.size() - 1] in ascii
        throw std::runtime_error("parse_npy_header: failed to read header");
    }

    // fortran order
    size_t loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True");

    // shape
    loc1 = header.find('(');
    size_t loc2 = header.find(')');
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    const std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    const bool littleEndian = (header[loc1] == '<' || header[loc1] == '|');
    if (!littleEndian) {
        throw std::runtime_error("parse_npy_header: only little endian data is supported");
    }

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find('\'');
    word_size = atoi(str_ws.substr(0, loc2).c_str());
    // std::cout << "word_size: " << word_size << std::endl;
}

auto load_the_npy_file(FILE* fp) -> npy::NpyArray
{
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    npy::parse_npy_header(fp, word_size, shape, fortran_order);

    npy::NpyArray arr(shape, word_size, fortran_order);
    const size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

auto npy::npy_load(const std::string& fname) -> npy::NpyArray
{
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp)
        throw std::runtime_error("npy_load: Unable to open file " + fname);

    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}

auto npy::load_npy_arr(const std::string& fname) -> std::tuple<std::unique_ptr<char[]>, size_t, size_t>
{
    FILE* fp = fopen(fname.c_str(), "rb");
    if (!fp)
        throw std::runtime_error("npy_load: Unable to open file " + fname);

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_npy_header(fp, word_size, shape, fortran_order);

    unsigned long num_vals = 1;
    for (const unsigned long i: shape)
        num_vals *= i;
    size_t n_bytes = word_size * num_vals;

    std::unique_ptr<char[]> arr(new char[n_bytes]);
    const size_t nread = fread(arr.get(), 1, n_bytes, fp);
    if (nread != n_bytes)
        std::cerr << "load_the_npy_file: failed fread" << std::endl;
    return std::make_tuple(std::move(arr), n_bytes, word_size);
}