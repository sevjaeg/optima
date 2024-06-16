/*
 * Adder and register instantiation for implementation
 */

 `include "timescale.v"
 
 module top #(
   parameter WIDTH = `WIDTH
 )(
   input      [WIDTH-1:0] a_i,
   input      [WIDTH-1:0] b_i,
   input                  valid_i,
   input                  clk_i,
   input                  rstn,
   output     [WIDTH-1:0] sum_o,
   output                 valid_o
 );
   
  wire [WIDTH-1:0] reg_in;
  wire [WIDTH-1:0] reg_out;
  wire             reg_valid_in;
  wire             reg_valid_out;

  adder #(
    .WIDTH(WIDTH)
  )
  i_adder (
    .a_i    (a_i),
    .b_i    (b_i),
    .c_o    (),
    .sum_o  (reg_in),
    .valid_i(valid_i),
    .valid_o(reg_valid_in)
  );

  register #(
  .WIDTH          (WIDTH),
  .VALID_IS_ENABLE(1)
  )
  u_register (
  .clk    (clk_i),
  .data_i (reg_in),
  .data_o (reg_out),
  .rstn   (rstn),
  .valid_i(reg_valid_in),
  .valid_o(reg_valid_out)
  );

  assign sum_o   = reg_out;
  assign valid_o = reg_valid_out;

 endmodule
 