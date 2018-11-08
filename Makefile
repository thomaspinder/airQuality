BIN:=src/utils

all:
		make init
		make clean

clean:
		rm -r src/data/*

init:
		make -C $(BIN)/
		