SRC				 := src
OBJ 			 := obj
BIN        := bin
MISC			 := ../cube/src

MKL_INC 	 := -m64 -I${MKLROOT}/include
INC        := -I$(MISC) $(MKL_INC)
MKL_LD		 := -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

CXX 			 := c++
LINKER     := $(CXX)
DEFS  		 := -DGEMM_L3_CACHE_SIZE=1940000
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