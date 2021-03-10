SRC				 := src
OBJ 			 := obj
BIN        := bin

MKL_INC 	 := -m64 -I${MKLROOT}/include
INC        := $(MKL_INC)
MKL_LD		 := -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

CC 				 := c++
LINKER     := $(CC)
DEFS  		 := -DGEMM_L3_CACHE_SIZE=1800000
CFLAGS     := -O3 -march=native -Wall -std=c++17 $(INC)
LDFLAGS    := $(MKL_LD) -lm -ldl -lpthread

SOURCES 	 := $(wildcard $(SRC)/*.cpp)
BINARIES 	 := $(patsubst $(SRC)/%.cpp, $(BIN)/%.x, $(SOURCES))

default: all
all: $(BINARIES)

#$(OBJ)/%.o: #$(SRC)/timer/%.c
#	$(CC) $(CFLAGS) -cpp $< -o $@

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CC) $(CFLAGS) $(DEFS) -c $< -o $@

$(BIN)/%.x: $(OBJ)/%.o #$(OBJ)/timer.o
	$(LINKER) $< $(LDFLAGS) -o $@ #$(OBJ)/timer.o

clean:
	rm -f $(BIN)/* $(OBJ)/*
