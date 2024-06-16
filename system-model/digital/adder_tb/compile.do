if {[file isdirectory work]} {file delete -force work};
valib work

set files [open "input.files" r]
while {[ gets $files line] >=0} {
    vlog -work work +incdir+../../include $line
}
