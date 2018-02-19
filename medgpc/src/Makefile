CC = icpc
CFLAGS = -mkl -qopenmp -simd -std=c++11 -O3 -qopt-report-phase=vec -qopt-report=2

JSON_LIB_DIR = /home/lifangc/install/rapidjson/include
TOP_SRC_DIR = /home/lifangc/workspace/release/MedGP/medgpc/src
OBJ_DIR = .

SRCS = $(wildcard $(TOP_SRC_DIR)/*/*.cpp)
OBJS = $(patsubst $(TOP_SRC_DIR)/*/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

TRAIN_MAIN = main_one_train.cpp
TRAIN_EXEC = main_one_train.o

TEST_MAIN = main_one_test.cpp
TEST_EXEC = main_one_test.o

all: train test

test: $(OBJS)
	$(CC) -I$(TOP_SRC_DIR) -I$(JSON_LIB_DIR) -o $(TEST_EXEC) $(OBJS) $(TEST_MAIN) $(CFLAGS)

train: $(OBJS)
	$(CC) -I$(TOP_SRC_DIR) -I$(JSON_LIB_DIR) -o $(TRAIN_EXEC) $(OBJS) $(TRAIN_MAIN) $(CFLAGS)

clean:
	rm -f *.o $(OBJ_DIR)/*.o
