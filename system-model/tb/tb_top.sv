`include "timescale.v"
`include "util.v"
`include "defines.v"

/*
 * When the Makefile is used for simulation, the parameters defined there are used
 */

module test #(
  parameter real VDD       = 1.0,
  parameter real TEMP      = 25.0,
  parameter bit MISMATCH   = 0,
  parameter bit AID        = 0,
  parameter bit MAC_ANALOG = 0, // if 0: digital (after ADC), otherwise analog
  parameter SRAM_WIDTH     = 4,
  parameter DAC_WIDTH      = SRAM_WIDTH,
  parameter ADC_WIDTH      = 2*SRAM_WIDTH,
  parameter ADDER_WIDTH    = ADC_WIDTH+3,
  parameter T0             = 0.2,  // [ns], 2^(SRAM_WIDTH-1)*T0 <= 2.3 ns to achieve 6 ns clock frequency
  parameter DAC_V_FS       = 1,
  parameter DAC_V_ZERO     = `V_TH
);


  localparam SRAM_ROWS     = 32;
  localparam MAC_LEN       = 10;
  localparam MC_SAMPLES    = 100;

  // +1 to have an address line even in the case of a single row
  localparam SRAM_ADDR_WIDTH = $clog2(SRAM_ROWS) + 1;

  localparam SEED         =  42;
  localparam DELTA_TIME   = 0.05;

  localparam CLOCK_PERIOD  = 6;  // do not change, 6 ns correspond to 166.7 MHz

  localparam string CSV_NAME = $sformatf("../out/operations.csv");

  typedef struct {
    logic [                  7 : 0] op;
    logic [SRAM_ADDR_WIDTH - 1 : 0] addr;
    logic [SRAM_WIDTH      - 1 : 0] data_sram;
    logic [DAC_WIDTH       - 1 : 0] data_dac;
    logic [                 63 : 0] adc_voltage;
    logic [ADDER_WIDTH     - 1 : 0] result;
    logic                           done;
    logic [                 63 : 0] energy_cell;
    logic [                 63 : 0] energy_adc;
    logic [                 63 : 0] energy_dac;
    logic [                 63 : 0] energy_dig;
  } tlm_t;

  logic clk, rstn;
  logic clock_running;

  logic test_running = 0;
  int cycle_count = 0;

  tlm_t tlm_data;
  int   file_descriptor, file_existed;

  int   test_count;
  int   product;

  real total_energy_cell = 0.0;
  real total_energy_adc  = 0.0;
  real total_energy_dac  = 0.0;
  real total_energy_dig  = 0.0;

  // Clock generation
  initial begin
    clk           = '0;
    clock_running = '1;
    while(clock_running) begin
      #(0.5*CLOCK_PERIOD) clk = (clk !== 1);
    end
    $finish;
  end

  // Reset generation
  initial begin
    rstn = '0;
    repeat(2) @(posedge clk);
    rstn = '1;
  end

  // File I/O
  initial begin
    file_existed = $fopen(CSV_NAME, "r");

    file_descriptor = $fopen(CSV_NAME, "a");
    if(!file_descriptor) begin
      $warning("%7t ps: File %s not opened", $realtime, CSV_NAME);
    end

    if(!file_existed) begin
      $fdisplay(file_descriptor, "vdd,temperature,mismatch,aid,mac_analog,bw_sram,bw_adc,bw_dac,bw_add,t0,dac_v0,dac_vfs,op,a,b,result,ref_result,energy cell,energy adc,energy dac,energy dig,vadc");
    end

    wait (clock_running == '0);
    $fclose(file_descriptor);
  end

  // Clock cycle count
  always @(posedge clk) begin
      if (test_running) begin
          cycle_count <= cycle_count + 1;
      end
  end

  // Test vectors
  int addrs   [0:MAC_LEN-1] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int weights [0:MAC_LEN-1] = { 5, 0, 13,  8, 1,  4, 15, 11, 3, 6};
  int inputs  [0:MAC_LEN-1] = {15, 0,  3, 12, 7, 14, 15,  2, 5, 9};

  // Test sequence
  initial begin
    ResetTLMObject;
    while(!rstn) @(posedge clk);

    if(MISMATCH) begin
      test_count = MC_SAMPLES;
    end else begin
      test_count = 1;
    end

    test_running = 1;
    $display("%7t ns: Starting tests at %m", 1000*$realtime);

    // Add(.a(5), .b(12), .debug(1), .sum(product), .energy(total_energy_dig));
    // Write(.write_data(0), .addr(1), .debug(1), .energy(total_energy_cell), .file_descriptor(file_descriptor));
    // Write(.write_data(15), .addr(2), .debug(1), .energy(total_energy_cell), .file_descriptor(file_descriptor));

    // Write(.write_data(0), .addr(0), .debug(1), .energy(total_energy_cell), .file_descriptor(file_descriptor));
    // Mul(.dac_data(15), .addr(0), .debug(1), .product(product), .energy_cell(total_energy_cell),
    //             .energy_adc(total_energy_adc), .energy_dac(total_energy_dac), .ref_b(15));

    // for (int i=0; i<MAC_LEN; i++) begin
    //   Write(.write_data(weights[i]), .addr(addrs[i]), .debug(1), .energy(total_energy_cell), .file_descriptor(file_descriptor));
    // end
    // MultiplyAccumulate(.dac_data(inputs), .addr(addrs), .debug(1), .accumulated_product(product), .energy_cell(total_energy_cell), .energy_adc(total_energy_adc), .energy_dac(total_energy_dac), .energy_dig(total_energy_dig));

    repeat(test_count) begin
      for (int i=0; i<2**SRAM_WIDTH; i++) begin
        Write(.write_data(i), .addr(0), .debug(0), .energy(total_energy_cell), .file_descriptor(file_descriptor));
        for (int j=0; j<2**DAC_WIDTH; j++) begin
          Mul(.dac_data(j), .addr(0), .debug(0), .product(product), .energy_cell(total_energy_cell),
              .energy_adc(total_energy_adc), .energy_dac(total_energy_dac), .ref_b(i));
          // $fdisplay(file_descriptor, "%0d,%0d,%0d,%0f,%0f,%e,%e,%e,%e,0.0", i, j, product, VDD, TEMP, energy_cell, energy_adc, energy_dac);
        end
      end
    end

    // repeat(test_count) begin
    //  //Write(.write_data(15), .addr(0), .debug(0), .energy(total_energy_cell), .file_descriptor(file_descriptor));
    //  Write(.write_data(15), .addr(0), .debug(0), .energy(total_energy_cell), .file_descriptor(file_descriptor));
    //  Mul(15, .addr(0), .debug(0), .product(product), .energy_cell(total_energy_cell),
    //            .energy_adc(total_energy_adc), .energy_dac(total_energy_dac), .ref_b(225));
    //   end

    $display("%7t ps: Test finished\nData written to %s", $realtime, CSV_NAME);
    test_running = 0;
    $display("\n\nTotal runtime %0e ns (%0d cycles)", cycle_count*CLOCK_PERIOD, cycle_count);
    $display("\n\nTotal energy cell %.2f pJ", 1e12*total_energy_cell);
    $display("Total energy ADC %.2f pJ", 1e12*total_energy_adc);
    $display("Total energy DAC %.2f pJ", 1e12*total_energy_dac);
    $display("Total energy digital adder %.2f pJ", 1e12*total_energy_dig);

    clock_running = '0;
  end

  ms_top #(
    .VDD            (VDD),
    .ADDER_WIDTH    (ADDER_WIDTH),
    .ADC_BIT_WIDTH  (ADC_WIDTH),
    .DAC_BIT_WIDTH  (DAC_WIDTH),
    .DAC_V_FS       (DAC_V_FS*VDD),
    .DAC_V_ZERO     (DAC_V_ZERO*VDD),
    .SRAM_WIDTH     (SRAM_WIDTH),
    .SRAM_ROWS      (SRAM_ROWS),
    .SRAM_ADDR_WIDTH(SRAM_ADDR_WIDTH),
    .DELTA_TIME     (DELTA_TIME),
    .T0             (T0),
    .AID            (AID),
    .SEED           (SEED),
    .TEMPERATURE    (TEMP),
    .MISMATCH       (MISMATCH),
    .MAC_ANALOG     (MAC_ANALOG)
  )
  i_ms_top (
    .clk        (clk),
    .rstn       (rstn),
    .addr_i     (tlm_data.addr),
    .data_dac_i (tlm_data.data_dac),
    .data_sram_i(tlm_data.data_sram),
    .op_i       (tlm_data.op),
    .data_o     (tlm_data.result),
    .done_o     (tlm_data.done),
    .array_energy_o (tlm_data.energy_cell),
    .adc_energy_o   (tlm_data.energy_adc),
    .dac_energy_o   (tlm_data.energy_dac),
    .dig_energy_o   (tlm_data.energy_dig),
    .v_adc_o        (tlm_data.adc_voltage)
  );

  /*
   * Utility functions
   */

  function logic [SRAM_ADDR_WIDTH-1:0] truncate_addr(input int addr);
    verify_addr(addr);
    if (SRAM_ROWS == 1) begin
      truncate_addr = '0;
    end else begin
      truncate_addr[SRAM_ADDR_WIDTH-1] = '0;
      if (SRAM_ROWS > 1) begin
        truncate_addr = addr[SRAM_ADDR_WIDTH-2:0];
      end
    end
  endfunction : truncate_addr

  function logic [SRAM_WIDTH-1:0] truncate_sram(input int data);
    verify_sram_data(data);
    truncate_sram = data[SRAM_WIDTH-1:0];
  endfunction : truncate_sram

  function logic [DAC_WIDTH-1:0] truncate_dac(input int data);
    verify_dac_data(data);
    truncate_dac = data[DAC_WIDTH-1:0];
  endfunction : truncate_dac

  function void verify_addr(input int addr);
    if ($unsigned(addr) > SRAM_ROWS-1) begin
      $warning("SRAM write addr %3x larger than SRAM block. The MSB(s) are truncated.", addr);
    end
  endfunction : verify_addr

  function void verify_sram_data(input int data);
    if ($unsigned(data) > 2**SRAM_WIDTH -1) begin
      $warning("SRAM write data %3x larger than SRAM row. The MSB(s) are truncated.", data);
    end
  endfunction : verify_sram_data

  function void verify_dac_data(input int data);
    if ($unsigned(data) > 2**DAC_WIDTH -1) begin
      $warning("DAC write data %3x larger than DAC range. The MSB(s) are truncated.", data);
    end
  endfunction : verify_dac_data

  /*
   * Test bench tasks (TLM)
   */

  task Write(
    input  int write_data,
    input  int addr,
    input      debug,
    inout real energy,
    input int  file_descriptor
  );
    real write_en, en_op;
  begin
    wait(tlm_data.done == 1'b0);

    tlm_data.op        = `OP_WRITE;
    tlm_data.data_sram = truncate_sram(write_data);
    tlm_data.addr      = truncate_addr(addr);

    wait(tlm_data.done == 1'b1);
    tlm_data.op = `OP_NOP;
    en_op = $bitstoreal(tlm_data.energy_cell);
    energy = energy + en_op;

    $fdisplay(file_descriptor, "%f,%f,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%e,%f,%f,%0d,%0d,%0d,%0d,%0d,%e,%e,%e,%e,%f",
              VDD, TEMP, MISMATCH, AID, MAC_ANALOG, SRAM_WIDTH, ADC_WIDTH, DAC_WIDTH, ADDER_WIDTH, T0, DAC_V_ZERO, DAC_V_FS, `OP_WRITE, write_data, 0, 0, 0, en_op, 0.0, 0.0, 0.0, 0.0);

    if (debug) $display("%7t ps: WR : *(%0d) = %0d @%.2f pJ", $realtime, tlm_data.addr, tlm_data.data_sram, 1e12*en_op);

  end
  endtask


  task Add(
    input  int a,
    input  int b,
    input      debug,
    output int sum,
    inout real energy
  );
  real en_op;
  begin
    wait(tlm_data.done == 1'b0);

    verify_sram_data(a);
    verify_dac_data(b);

    tlm_data.op        = `OP_ADD;
    tlm_data.data_sram = truncate_sram(a);
    tlm_data.data_dac  = truncate_dac(b);

    wait(tlm_data.done == 1'b1);
    tlm_data.op = `OP_NOP;

    en_op = $bitstoreal(tlm_data.energy_dig);
    energy = energy + en_op;
    sum = tlm_data.result;

    $fdisplay(file_descriptor, "%f,%f,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%e,%f,%f,%0d,%0d,%0d,%0d,%0d,%e,%e,%e,%e,%f",
              VDD, TEMP, MISMATCH, AID, MAC_ANALOG, SRAM_WIDTH, ADC_WIDTH, DAC_WIDTH, ADDER_WIDTH, T0, DAC_V_ZERO, DAC_V_FS, `OP_ADD, a, b, tlm_data.result, a+b, 0.0, 0.0, 0.0, en_op, 0.0);

    if (debug) $display("%7t ps: ADD: %0d + %0d = %0d @%.2f pJ", $realtime, tlm_data.data_sram, tlm_data.data_dac, tlm_data.result, 1e12*en_op);
  end
  endtask

  task Mul(
    input  int dac_data,
    input  int ref_b,
    input  int addr,
    input      debug,
    inout real energy_cell,
    inout real energy_adc,
    inout real energy_dac,
    output int product
  );
    real e_cell, e_adc, e_dac, e_dig, total_e, v_adc;
  begin
    wait(tlm_data.done == 1'b0);

    tlm_data.op       = `OP_MUL;
    tlm_data.addr     = truncate_addr(addr);
    tlm_data.data_dac = truncate_dac(dac_data);

    wait(tlm_data.done == 1'b1);
    tlm_data.op = `OP_NOP;
    product = tlm_data.result;
    v_adc = $bitstoreal(tlm_data.adc_voltage);
    e_cell = $bitstoreal(tlm_data.energy_cell);
    e_adc = $bitstoreal(tlm_data.energy_adc);
    e_dac = $bitstoreal(tlm_data.energy_dac);
    total_e = e_cell + e_adc + e_dac;

    energy_cell = energy_cell + e_cell;
    energy_adc = energy_adc + e_adc;
    energy_dac = energy_dac + e_dac;

    $fdisplay(file_descriptor, "%f,%f,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%e,%f,%f,%0d,%0d,%0d,%0d,%0d,%e,%e,%e,%e,%f",
              VDD, TEMP, MISMATCH, AID, MAC_ANALOG, SRAM_WIDTH, ADC_WIDTH, DAC_WIDTH, ADDER_WIDTH, T0, DAC_V_ZERO, DAC_V_FS, `OP_MUL, dac_data, ref_b, tlm_data.result, dac_data*ref_b, e_cell, e_adc, e_dac, 0.0, v_adc);

    if (debug) $display("%7t ps: MUL: %0d x *(%0d) = %0d @%.2f pJ", $realtime,
                                                                  tlm_data.data_dac, tlm_data.addr, tlm_data.result, 1e12*total_e);
  end
  endtask

  task MultiplyAccumulate(
    input  int dac_data [MAC_LEN-1:0],
    input  int addr     [MAC_LEN-1:0],
    input      debug,
    output int accumulated_product,
    inout real energy_cell,
    inout real energy_adc,
    inout real energy_dac,
    inout real energy_dig
  );
    int i, product;
    real e_cell, e_adc, e_dac, e_dig, total_e_cell, total_e_adc, total_e_dac, total_e_dig, total_e;
  begin
    total_e_cell = 0.0;
    total_e_adc = 0.0;
    total_e_dac = 0.0;
    total_e_dig = 0.0;

    if (~MAC_ANALOG) begin
      ResetReg;
    end

    for (int i=0; i<MAC_LEN; i++) begin
        if (MAC_ANALOG) begin
          MultiplyAccumulateAnalog(.dac_data(dac_data[i]), .addr(addr[i]), .debug(debug), .energy_cell(e_cell), .energy_dac(e_dac));
          if(i==MAC_LEN-1) begin
            MultiplyAccumulateAnalogCapture(.dac_data(dac_data[i]), .addr(addr[i]), .debug(debug), .accumulated_product(product), .energy_cell(e_cell), .energy_adc(e_adc), .energy_dac(e_dac));
            total_e_dac  = total_e_dac  + e_dac;
          end
          total_e_cell = total_e_cell + e_cell;
          total_e_dac  = total_e_dac  + e_dac;
        end else begin  // digital MAC
          MultiplyAccumulateDigital(.dac_data(dac_data[i]), .addr(addr[i]), .debug(debug), .accumulated_product(product), .energy_cell(e_cell), .energy_adc(e_adc), .energy_dac(e_dac), .energy_dig(e_dig));
          total_e_cell = total_e_cell + e_cell;
          total_e_adc  = total_e_adc  + e_adc;
          total_e_dac  = total_e_dac  + e_dac;
          total_e_dig  = total_e_dig  + e_dig;
        end
      end
      accumulated_product = product;
      energy_cell = energy_cell + total_e_cell;
      energy_adc = energy_adc + total_e_adc;
      energy_dac = energy_dac + total_e_dac;
      energy_dig = energy_dig + total_e_dig;

      total_e = total_e_cell + total_e_adc + total_e_dac + total_e_dig;

      if (MAC_ANALOG) begin
        $fdisplay(file_descriptor, "%f,%f,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%e,%f,%f,%0d,%0d,%0d,%0d,%0d,%e,%e,%e,%e,%f",
                  VDD, TEMP, MISMATCH, AID, MAC_ANALOG, SRAM_WIDTH, ADC_WIDTH, DAC_WIDTH, ADDER_WIDTH, T0, DAC_V_ZERO, DAC_V_FS, `OP_MAC_AN, 0, 0, product, 0, total_e_cell, total_e_adc, total_e_dac, total_e_dig,0.0);
      end else begin
        $fdisplay(file_descriptor, "%f,%f,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%e,%f,%f,%0d,%0d,%0d,%0d,%0d,%e,%e,%e,%e,%f",
                  VDD, TEMP, MISMATCH, AID, MAC_ANALOG, SRAM_WIDTH, ADC_WIDTH, DAC_WIDTH, ADDER_WIDTH, T0, DAC_V_ZERO, DAC_V_FS, `OP_MAC_DIG, 0, 0, product, 0, total_e_cell, total_e_adc, total_e_dac, total_e_dig,0.0);
      end

    if (debug) $display("%7t ps: MAC: sum %3d (%0d elements) @%.2f pJ", $realtime, product, MAC_LEN, 1e12*total_e);
  end
  endtask

  task MultiplyAccumulateDigital(
    input  int dac_data,
    input  int addr,
    input      debug,
    output int accumulated_product,
    inout real energy_cell,
    inout real energy_adc,
    inout real energy_dac,

    inout real energy_dig
  );
  begin
    wait(tlm_data.done == 1'b0);

    tlm_data.op       = `OP_MAC_DIG;
    tlm_data.addr     = truncate_addr(addr);
    tlm_data.data_dac = truncate_dac(dac_data);

    wait(tlm_data.done == 1'b1);
    tlm_data.op = `OP_NOP;
    accumulated_product = tlm_data.result;

    energy_cell = $bitstoreal(tlm_data.energy_cell);
    energy_adc = $bitstoreal(tlm_data.energy_adc);
    energy_dac = $bitstoreal(tlm_data.energy_dac);
    energy_dig = $bitstoreal(tlm_data.energy_dig);

    if (debug) $display("%7t ps: MAC result %3d with %2d and data from %2h", $realtime, tlm_data.result,
                                                                  tlm_data.data_dac, tlm_data.addr);
  end
  endtask

  task MultiplyAccumulateAnalog(
    input  int dac_data,
    input  int addr,
    input      debug,
    inout real energy_cell,
    inout real energy_dac
  );
  begin
    wait(tlm_data.done == 1'b0);

    tlm_data.op       = `OP_MAC_AN;
    tlm_data.addr     = truncate_addr(addr);
    tlm_data.data_dac = truncate_dac(dac_data);

    wait(tlm_data.done == 1'b1);
    tlm_data.op = `OP_NOP;

    energy_cell = $bitstoreal(tlm_data.energy_cell);
    energy_dac = $bitstoreal(tlm_data.energy_dac);

    if (debug) $display("%7t ps: Analog MAC with %2d and data from %2h", $realtime,
                                                                  tlm_data.data_dac, tlm_data.addr);
  end
  endtask

  task MultiplyAccumulateAnalogCapture(
    input  int dac_data,
    input  int addr,
    input      debug,
    output int accumulated_product,
    inout real energy_cell,
    inout real energy_adc,
    inout real energy_dac
  );
  begin
    wait(tlm_data.done == 1'b0);

    tlm_data.op       = `OP_MAC_AN_CAP;
    tlm_data.addr     = truncate_addr(addr);
    tlm_data.data_dac = truncate_dac(dac_data);
    product = tlm_data.result;

    wait(tlm_data.done == 1'b1);
    tlm_data.op = `OP_NOP;

    energy_cell = $bitstoreal(tlm_data.energy_cell);
    energy_adc = $bitstoreal(tlm_data.energy_adc);
    energy_dac = $bitstoreal(tlm_data.energy_dac);

    if (debug) $display("%7t ps: Analog MAC result %3d with %2d and data from %2h", $realtime, tlm_data.result,
                                                                  tlm_data.data_dac, tlm_data.addr);
  end
  endtask

  task ResetReg;
    begin
      wait(tlm_data.done == 1'b0);
      tlm_data.op = `OP_RST_REG;
      wait(tlm_data.done == 1'b1);
      tlm_data.op = `OP_NOP;
    end
    endtask

  task ResetTLMObject;
  begin
    tlm_data.op         = `OP_NOP;
    tlm_data.addr       = truncate_addr(0);
    tlm_data.data_dac   = truncate_dac(0);
    tlm_data.data_sram  = truncate_sram(0);
  end
  endtask

endmodule
