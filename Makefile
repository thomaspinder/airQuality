BIN:=src/utils

all:
		make init
		make clean
		make reqs

clean:
		rm -r src/data/*

init:
		pip install --user -r requirements.txt
		make -C $(BIN)/

reqs:
		sh src/utils/reqs_gen.sh
