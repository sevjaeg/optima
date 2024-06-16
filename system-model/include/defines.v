// testbench interface operation codes

/* verilog_format: off */

`define OP_NOP        8'd0
`define OP_WRITE      8'd1
`define OP_MUL        8'd2
`define OP_ADD        8'd3
`define OP_MAC_DIG    8'd4
`define OP_MAC_AN     8'd5
`define OP_MAC_AN_CAP 8'd6
`define OP_RST_REG    8'd7

// ms_top control states

`define S_IDLE       8'd0
`define S_WRITE      8'd1
`define S_MUL_DISC   8'd3
`define S_MUL_SAMPLE 8'd4
`define S_MUL_LATCH  8'd5
`define S_ADD_SUM    8'd6
`define S_ADD_LATCH  8'd7
`define S_ACC_AN     8'd8
`define S_RST        8'd9

// ms_top register driver

`define REG_DRIVER_IDLE  8'd0
`define REG_DRIVER_ADC   8'd1
`define REG_DRIVER_ADD   8'd2
`define REG_DRIVER_MAC   8'd3

// ms_top adder driver

`define ADD_DRIVER_IDLE  8'd0
`define ADD_DRIVER_MAC   8'd1
`define ADD_DRIVER_EXT   8'd2

// ms_top ACC driver

`define ADC_DRIVER_IDLE  0
`define ADC_DRIVER_CS    1
`define ADC_DRIVER_ACC   2

// nominal SRAM conditions

`define NOMINAL_VDD   1.0
`define NOMINAL_TEMP 25.0
`define V_TH          0.3

/* verilog_format: on */
