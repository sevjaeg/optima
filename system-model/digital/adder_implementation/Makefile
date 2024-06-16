DESIGN := adder

LOG_DIR := logs
OUT_DIR := out
REPORT_DIR := reports
TMP_DIR := tmp

SYN_SCRIPT := genus.tcl
PNR_SCRIPT := $(SCRIPT_DIR)/implementation.tcl

all: synth

.PHONY: .synth-$(DESIGN).timestamp
.synth-$(DESIGN).timestamp:
	genus -batch -log ./$(LOG_DIR)/$(DESIGN)_genus -files $(SYN_SCRIPT)

synth: logdir .synth-$(DESIGN).timestamp

logdir:
	mkdir -p $(LOG_DIR)

clean-tools:
	rm -rf genus.* innovus.* .cadence
	rm -f *.old
	rm -f scheduling_file.cts.*
	rm -rf .st_launch_*
	rm -f *.conn.rpt *.metalfill.rpt power.rpt *.geom.rpt
	rm -rf $(LOG_DIR) mp_data
	rm -rf genus2invs_*
	rm -rf rtl_*_db
	rm -rf timingReports checkDesign
	rm -rf .timing_file*
	rm -f voltus_power_missing_netnames.rpt
	rm -f .powerAnalysis.pinfo
	
clean: clean-tools	
	rm -rf fv fv*/
	rm -rf $(OUT_DIR) $(REPORT_DIR) $(TMP_DIR)
	rm -f .*.timestamp .*.tstamp

.PHONY: all clean clean-tools synth
