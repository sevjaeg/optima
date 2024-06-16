#!/usr/bin/bash

voltages=(1.0) # (0.9, 0.95, 1.0, 1.05, 1.1)
temperatures=(25) # (0.0, 25.0, 50.0, 75.0)
mismatches=(1)
aids=(0)
t0s=(0.16 0.2 0.24 0.28)
dacv0s=(0.3 0.4 0.5 )
dacvfss=(0.7 0.8, 0.9 1.0 )

rm ../../data-analysis/data/operations.csv

make compile

start_time=`date +%s`

for vdd in "${voltages[@]}"; do
    for temp in "${temperatures[@]}"; do
        for mismatch in "${mismatches[@]}"; do
            for aid in "${aids[@]}"; do
                for t0 in "${t0s[@]}"; do
                    for dacv0 in "${dacv0s[@]}"; do
                        for dacvfs in "${dacvfss[@]}"; do
                            echo "VDD:$vdd T:$temp Mismatch:$mismatch AID:$aid T0:$t0 DAC V_0:$dacv0 DAC V_FS:$dacvfs"
                            make VDD="$vdd" TEMPERATURE="$temp" MISMATCH="$mismatch" AID="$aid" T0="$t0" DAC_V_ZERO="$dacv0" DAC_V_FS="$dacvfs" run
                        done
                    done
                done
            done
        done
    done
done

end_time=`date +%s`
echo execution time `expr $end_time - $start_time` s.
