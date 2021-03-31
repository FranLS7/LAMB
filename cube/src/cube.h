#ifndef GEMM_CUBE
#define GEMM_CUBE

namespace lamb{

  struct Axis{
    int min_size;
    int max_size;
    int npoints;
    int *points;
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

      // Function that returns a certain value from the eff cube
      double get_value (const int d1, const int d2, const int d3) const;

      // double get_data (const int i) const;

      // int get_axis_value (const int ax, const int i) const;


    private:
      // Attributes
      double *data;

      int total_points;

      Axis *axes[3];

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
      bool is_in_cube (const int *dims, int *indices) const;

      // Function that extracts one value that we know already exists in the cube.
      // Basically, this function summarises a set of coordinates' translations.
      double access_cube (const int *indices) const;

      // Function that gets the indices of the points containing the original point
      // we have got to interpolate.
      void get_ranges (const int *dims, const int *indices, int *ranges) const;

      double trilinear_inter (const int *dims_o, const int *ranges) const;
  };
}

#endif








