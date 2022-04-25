SRC				 := benchmark
OBJ 			 := build
BIN        := bin
MISC			 := src

MKL_INC 	 := -m64 -I${MKLROOT}/include
INC        := -I$(MISC) $(MKL_INC)
MKL_LD		 := -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

CXX 			 := c++
LINKER     := $(CXX)
DEFS  		 := -DGEMM_L3_CACHE_SIZE=5000000 -DBENCH_REPS=10 -DN_THREADS=10 -DMARGIN_AN=0.10 -DRATIO_AN=0.7 -DTHRESHOLD=500#3194880
CFLAGS     := -O3 -fopenmp -march=native -Wall -std=c++17 $(INC)
OMP				 := -fopenmp
LDFLAGS    := $(MKL_LD) -lm -ldl -lpthread

SOURCES 	 := $(wildcard $(SRC)/*.cpp)
BINARIES 	 := $(patsubst $(SRC)/%.cpp, $(BIN)/%.x, $(SOURCES))

default: all
all: $(BINARIES)

$(OBJ)/%.o: $(MISC)/%.cpp $(MISC)/%.h # $(SRC)/cube/%.cpp $(SRC)/cube/%.h
	$(CXX) $(CFLAGS) $(OMP) $(DEFS) -c $< -o $@

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) $(CFLAGS) $(DEFS) -c $< -o $@

$(BIN)/%.x: $(OBJ)/%.o $(OBJ)/cube_old.o $(OBJ)/common.o $(OBJ)/MC4.o $(OBJ)/anomalies.o $(OBJ)/MCX.o $(OBJ)/exploration.o $(OBJ)/operation.o
	$(LINKER) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(BIN)/* $(OBJ)/*