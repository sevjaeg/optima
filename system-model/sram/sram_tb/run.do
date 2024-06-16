vasim work.test -t ps -outpath out
do wave.do
restart -force

run 500 ns
wave zoom full
