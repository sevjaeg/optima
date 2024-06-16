`include "timescale.v"
`include "util.v"

module adder_tb;
  localparam CLOCK_PERIOD = 10;  // 100 MHz

  localparam ADDER_WIDTH = 8;

  logic clk;
  logic rstn;

  integer sum = 0;

  logic [ADDER_WIDTH-1:0]  adder_a       = 0;
  logic [ADDER_WIDTH-1:0]  adder_b       = 0;
  logic                    adder_valid_i = 0;
  logic [ADDER_WIDTH-1:0]  adder_sum;
  logic                    adder_carry;
  logic                    register_valid_o;
  logic [ADDER_WIDTH-1:0]  register_sum;
  
  // Clock generation
  initial begin
    clk = 0;
    forever begin
      #(0.5*CLOCK_PERIOD) clk = (clk !== 1);
    end
  end
  
  // Test case sequence
  initial begin
    rstn = 0;
    repeat(2) @(posedge clk);
    $display("%7t ps: Starting tests", $realtime);
    rstn = 1;

    Add(4, 6, sum);
    $display("%7t ps: sum: %3d", $realtime, sum);

    Add(127, 127, sum);
    $display("%7t ps: sum: %3d", $realtime, sum);

    Add(128, 128, sum);
    $display("%7t ps: sum: %3d", $realtime, sum);

    repeat(10) @(posedge clk);
    $display("%7t ps: Finishing simulation", $realtime);
    $finish;
  end

  adder #(
    .WIDTH(ADDER_WIDTH)
  ) i_adder (
    .a_i(adder_a),
    .b_i(adder_b),
    .sum_o(adder_sum),
    .c_o(adder_carry)
  );
    
  register #(
    .WIDTH(ADDER_WIDTH)
  ) i_register (
    .clk    (clk),
    .data_i (adder_sum),
    .data_o (register_sum),
    .rstn   (rstn),
    .valid_i(adder_valid_i),
    .valid_o(register_valid_o)
  );

  /*
   * Utility tasks
   * All tasks are blocking, i.e. they return once the operation is finished
   * Thus they do not check if an earlier operation has finished
   */

  task Add(
    input  integer a,
    input  integer b,
    output integer sum
  );
  begin
    $display("%7t ps: Adding  %3d to %3d", $realtime, a, b);
    if(~rstn) begin
      @(posedge rstn);
    end
    
    if(register_valid_o) begin
      @(negedge register_valid_o);
    end
    
    if ($unsigned(a) > 2**ADDER_WIDTH -1) begin
      $warning("Adder input a %x larger than adder width. The MSB(s) are truncated.", a);
    end
    if ($unsigned(b) > 2**ADDER_WIDTH -1) begin
      $warning("Adder input b %x larger than adder width. The MSB(s) are truncated.", b);
    end

    adder_a = a[ADDER_WIDTH-1:0];
    adder_b = b[ADDER_WIDTH-1:0];
    adder_valid_i = 1;

    @(posedge register_valid_o);
    if (adder_carry) begin
      $warning("Overflow in addition");
    end
    // This only works up to 32 bits
    sum <= $unsigned(register_sum);
    adder_valid_i = 0;
    `NextCycle(clk, 0.1*CLOCK_PERIOD)
  end
  endtask

endmodule
