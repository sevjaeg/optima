/*
 * Behavioural combinational adder model
 */

`include "timescale.v"

module adder #(
  parameter WIDTH = 8
)(
  input      [WIDTH-1:0] a_i,
  input      [WIDTH-1:0] b_i,
  input                  valid_i,
  output     [WIDTH-1:0] sum_o,
  output                 valid_o,
  output                 c_o,
  output reg [     63:0] energy_o
);

  /*
   * Delay model: Constant combinational delay
   */
  localparam DELAY = 0.5;
  
  localparam [WIDTH:0] zero_val = 0;

  wire [WIDTH:0] sum;

  always @* begin
      if(valid_i) begin
        energy_o = $realtobits(1e-12*(0.00045606*WIDTH**2 + 0.04853065*WIDTH + 0.04391967));
      end else begin
        energy_o = 0.0;
      end
  end

  assign sum   = valid_i ? a_i + b_i : zero_val;

  assign #(DELAY) sum_o   = sum[WIDTH-1:0];
  assign #(DELAY) c_o     = sum[WIDTH];

  assign #(DELAY) valid_o = valid_i;

endmodule
