`define assert(signal, value) \
  if (signal !== value) begin \
    $error("ERROR %m: %x != %x", signal, value); \
  end

`define NextCycle(clk, wait_time) \
  @(posedge clk) \
  #(wait_time);

`define AddrWidth(rows) \
  $clog2(rows) + 1
