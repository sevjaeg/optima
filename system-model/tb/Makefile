all: cli

VDD         = 1.0
TEMPERATURE = 25.0
MISMATCH    = 0
AID         = 0
T0          = 0.2
DAC_V_ZERO  = 0.3
DAC_V_FS    = 1.0

.PHONY: run
run: cli

.PHONY: compile
compile: outdir
	vasim -c -do "do compile.do; quit" -outpath out

.PHONY: gui
gui: compile
	vasim -i -do "do run.do ${VDD} ${TEMPERATURE} ${MISMATCH} ${AID} ${T0} ${DAC_V_ZERO} ${DAC_V_FS}"

.PHONY: cli
cli: compile
	vasim -c -do "do run.do ${VDD} ${TEMPERATURE} ${MISMATCH} ${AID} ${T0} ${DAC_V_ZERO} ${DAC_V_FS}; quit"

.PHONY: outdir
outdir:
	mkdir -p ../out

.PHONY: clean
clean:
	rm -rf work out ../out
	rm -f transcript* modelsim.ini
	rm -f vsim.wlf
	rm -rf wlft* *.vstf
	rm -f *.log .*.tuwien.ac.at
	rm -f test.cir test_*
	rm -f *.error
