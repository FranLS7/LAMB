#ifndef GEMM_CUBE_CLASS
#define GEMM_CUBE_CLASS

#include <fstream>
#include <string>
#include <vector>

namespace cube{

struct Axis{
  int min_size;
  int max_size;
  int npoints;
  std::vector<int> points;
};


class GEMM_Cube{
  public:
    GEMM_Cube();

    // Class constructor to which a filename (.csv) is passed so the
    // instance is automatically loaded into memory.
    GEMM_Cube (std::string filename, const int overhead);

    // Function that loads a .csv file into the cube and creates all the
    // internal attributes. This function assumes the file and the cube are
    // both linearised.
    // overhead : number of characters to be discarded in the file's headers.
    bool load_cube (std::string filename, const int overhead);

    // Function that returns a certain value from the cube.
    double get_value (const int d1, const int d2, const int d3) const;

    // Function that shows the cube's information on screen.
    void print_info ();

    // double get_data (const int i) const;

    // int get_axis_value (const int ax, const int i) const;

    /**
     * Adds the headers to an output file for a generated cube. Format:
     * # nthreads
     * # d0:    min_size, max_size, npoints
     * # ...
     * # di-1:  min_size, max_size, npoints
     * # d0:    points
     * # ...
     * # di-1:  points
     * # | ndim dims || nsamples samples |
     *
     * @param ofile     Output file manager which has been previously opened.
     * @param ndim      The number of dimensions in the cube (usually 3).
     * @param min_size  Allocated memory with the min sizes for each dimension.
     * @param max_size  Allocated memory with the max sizes for each dimension.
     * @param npoints   Allocated memory with the number of points for each
     *      dimension (might be different depending on the dimension).
     * @param nsamples  The number of samples to store in the output file.
     * @param nthreads  The number of threads to use in the computation - is stored.
     * @param points    Allocated memory with the points for each dimension (might
     *      be different depending on the dimension).
     */
    void print_header_cube (std::ofstream &ofile, const int ndim, const int* min_size, 
        const int* max_size, const int* npoints, const int nsamples, const int nthreads, 
        const int **points) const;


  private:
    // Attributes
    std::vector<double> data;

    int total_points;

    std::vector<Axis> axes;

    Axis d0, d1, d2;

    // ==================================================================
    //   - - - - - - - - - - - Internal functions - - - - - - - - - - -
    // ==================================================================

    // Creating this list of points would be useful when the sampling is not uniform
    void create_points ();

    // Function that determines whether a certain point is in the cube.
    // iterate over the different dimensions and check
    // whether all of them are in the list of points. We use the function
    // binarySearch in this case for all the dimensions, because we assume that
    // all the dimensions take the same values in the cube (uniformly spaced)
    bool find (std::vector<int> dims_o, std::vector<int> indices) const;

    // Function that extracts one value that we know already exists in the cube.
    // Basically, this function summarises a set of coordinates' translations.
    double access_cube (std::vector<int> indices) const;

    // Function that gets the indices of the points containing the original point
    // we have got to interpolate.
    std::vector<std::vector<int>> get_ranges (std::vector<int> dims_o, 
        std::vector<int> indices) const;

    double trilinear_inter (std::vector<int> dims_o, 
        std::vector<std::vector<int>> ranges) const;
};
} // namespace lamb

#endif








