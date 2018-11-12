BIN:=src/utils

all:
		make reqs
		make init
		make prep

init:
		pip install --user -r requirements.txt
		make -C $(BIN)/

reqs:
		sh src/utils/reqs_gen.sh

prep:
		python src/modelling/modelling.py

rtest:
		Rscript src/modelling/integrate.R