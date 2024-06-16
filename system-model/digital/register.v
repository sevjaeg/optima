/*
 * Behavioural adder model with generic bit width and single-cycle delay
 */

module register #(
    parameter WIDTH = 8,
    parameter VALID_IS_ENABLE = 0
) (
    input              clk,
    input              rstn,
    input  [WIDTH-1:0] data_i,
    input              valid_i,
    output [WIDTH-1:0] data_o,
    output             valid_o
);

  reg [WIDTH-1:0] data_r;
  reg             valid_r;

  always @(posedge (clk)) begin
    if (!rstn) begin
      data_r  <= 0;
      valid_r <= 1'b0;
    end else begin

      if (VALID_IS_ENABLE) begin : gen_enable
        if (valid_i) begin
          data_r  <= data_i;
          valid_r <= valid_i;
        end else begin
          data_r  <= data_r;
          valid_r <= valid_r;
        end
      end else begin
        data_r  <= data_i;
        valid_r <= valid_i;
      end
    end
  end

  assign data_o  = data_r;
  assign valid_o = valid_r;

endmodule
