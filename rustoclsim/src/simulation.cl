// Completely by-the-book reference implementation of xorshift
uint xorshift32(__global uint* state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Select an item at random from a buffer
uint random_select(__global uint* state, __global uint* precomp, uint len) {
    return precomp[xorshift32(state) % len];
}

__kernel void ocl_simulate_demand(
    __global uint* seed,
    __global uint* job_lot_zipf_precomp,
    __global uint* itemwise_traffic_zipf_precomp,
    __global ulong* all_successful_transactions,
    __global ulong* all_successful_sales,
    __global ulong* all_failed_transactions,
    __global ulong* all_failed_sales,
    float starting_quantity,
    uint lead_time,
    int safety_stock,
    int order_quantity,
    uint precomp_size
) {
    int me = get_global_id(0);
    ulong successful_transactions = 0;
    ulong successful_sales = 0;
    ulong failed_transactions = 0;
    ulong failed_sales = 0;
    int stock = starting_quantity;
    uint trucks[10];


    for (int day=0; day<365; day++) {
        // A truck arrived
        stock += trucks[day % lead_time];
        // This many customers arrive
        uint customer_count = random_select(&seed[me], itemwise_traffic_zipf_precomp, precomp_size);
        for (uint _customer=0; _customer < customer_count; _customer++) {
            // This customer wants this many
            int request = random_select(&seed[me], job_lot_zipf_precomp, precomp_size);
            if (stock >= request) {
                // There are enough.
                successful_transactions += 1;
                successful_sales += request;
                stock -= request;
            } else {
                // There are not enough
                failed_transactions += 1;
                failed_sales += request;
            }
        }
        // The day is over. Start making orders.
        if (stock < safety_stock) {
            int short_by = max(safety_stock - stock, 0);
            int orders = (short_by + order_quantity - 1) / order_quantity;
            trucks[(day + lead_time - 1) % lead_time] = orders * order_quantity;
        }
    }
    all_successful_transactions[me] = successful_transactions;
    all_successful_sales[me] = successful_sales;
    all_failed_transactions[me] = failed_transactions;
    all_failed_sales[me] = failed_sales;
}