all: cli

compile:
	vasim -c -do "do compile.do; quit" -outpath out

gui: compile
	vasim -i -do "run.do"

cli: compile
	vasim -c -do "do run.do; quit" 

clean:
	rm -rf work out
	rm -f transcript* modelsim.ini
	rm -f vsim.wlf
	rm -rf wlft* *.vstf
	rm -f *.log .*.tuwien.ac.at
	rm -f test.cir test_*
