SRC				 := src
OBJ 			 := obj
LIB        := lib

MKL_INC 	 := -m64 -I${MKLROOT}/include
INC        := $(MKL_INC)
MKL_LD		 := -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

CXX 			 := c++
LINKER     := $(CXX)
DEFS  		 := -DGEMM_L3_CACHE_SIZE=1800000
CFLAGS     := -O3 -march=native -Wall -std=c++17 $(INC)
LDFLAGS    := $(MKL_LD) -lm -ldl -lpthread
LIB_FLAG	 := -fPIC -shared

SOURCES 	 := $(wildcard $(SRC)/*.cpp)
OBJECTS 	 := $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(SOURCES))

default: all
all: $(LIB)/GC_lib_mt.dylib

$(OBJ)/%.o: $(SRC)/%.cpp # $(SRC)/%.h
	$(CXX) $(CFLAGS) $(DEFS) -c $< -o $@

$(LIB)/GC_lib_mt.dylib: $(OBJECTS)
	$(LINKER) $(LDFLAGS) $(LIB_FLAG) $^ -o $@
	rm $(OBJ)/*

clean:
	rm -f $(OBJ)/* $(LIB)/*