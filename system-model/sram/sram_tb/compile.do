if {[file isdirectory work]} {file delete -force work};
valib work

set files [open "input.files" r]
while {[ gets $files line] >=0} {
    vlog -work work +incdir+../../include $line
}

set files_analog [open "input-analog.files" r]
while {[ gets ${files_analog} line] >=0} {
    valog -work work +incdir+../../include $line
}