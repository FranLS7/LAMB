SRC				 := src
OBJ 			 := obj
BIN        := bin
MISC			 := ../cube/src

MKL_INC 	 := -m64 -I${MKLROOT}/include
INC        := -I$(MISC) $(MKL_INC)
MKL_LD		 := -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

CXX 			 := c++
LINKER     := $(CXX)
DEFS  		 := -DGEMM_L3_CACHE_SIZE=1800000
CFLAGS     := -O3 -march=native -Wall -std=c++17 $(INC)
LDFLAGS    := $(MKL_LD) -lm -ldl -lpthread

SOURCES 	 := $(wildcard $(SRC)/*.cpp)
BINARIES 	 := $(patsubst $(SRC)/%.cpp, $(BIN)/%.x, $(SOURCES))

default: all
all: $(BINARIES)

$(OBJ)/%.o: $(MISC)/%.cpp $(MISC)/%.h # $(SRC)/cube/%.cpp $(SRC)/cube/%.h
	$(CXX) $(CFLAGS) $(DEFS) -c $< -o $@

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) $(CFLAGS) $(DEFS) -c $< -o $@

$(BIN)/%.x: $(OBJ)/%.o $(OBJ)/cube.o $(OBJ)/common.o   # $(OBJ)/cube.o $(OBJ)/common.o
	$(LINKER) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(BIN)/* $(OBJ)/*