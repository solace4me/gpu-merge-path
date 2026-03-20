# ============================================================
# Makefile for GPU Merge-Path CUDA Project
# ============================================================

# CUDA compiler
NVCC = nvcc

# Target executable name
TARGET = merge_path

# Source file
SRC = merge_path.cu

# Compiler flags
NVCC_FLAGS = -O3 -std=c++11

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET) results.txt