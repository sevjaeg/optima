vasim -GVDD=$1 -GTEMP=$2 -GMISMATCH=$3 -GAID=$4 -GT0=$5 -GDAC_V_ZERO=$6 -GDAC_V_FS=$7 work.test -t ps -outpath out
do wave.do
restart -force

run 8000000 ns
wave zoom full
