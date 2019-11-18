#!/usr/bin/env python3
import numpy as np
cimport numpy as np
import random

cdef class Simulation:
    ''' Simulates the demand of one product over time
    
        Safety stock is how many items to target on the shelf at all times.
        Lead time is how many days between making an order and when it arrives on the shelf.
        Order quantity is how many items you can order at a time (orders are an integer multiple of this)
        Job lot zipf is the zipf parameter of the distribution of job lot sizes - how many people buy at a time
        Itemwise traffic zipf is analogous, but for the distribution of customer traffic for that item
    '''
    cdef int safety_stock
    cdef int lead_time
    cdef int order_quantity
    cdef double job_lot_zipf
    cdef double itemwise_traffic_zipf
    cdef np.ndarray job_lot_zipf_precomp
    cdef np.ndarray itemwise_traffic_zipf_precomp

    def __init__(self, int safety_stock, int lead_time, int order_quantity, double job_lot_zipf=2.75, double itemwise_traffic_zipf=4.0):
        self.safety_stock = safety_stock
        self.lead_time = lead_time
        self.order_quantity = order_quantity
        self.job_lot_zipf = job_lot_zipf
        self.itemwise_traffic_zipf = itemwise_traffic_zipf
        self.job_lot_zipf_precomp = np.random.zipf(self.job_lot_zipf, 16 << 20)
        self.itemwise_traffic_zipf_precomp = np.random.zipf(self.itemwise_traffic_zipf, 16 << 20)
    

    cpdef simulate_demand(self, int starting_quantity):
        cdef int successful_transactions = 0
        cdef int successful_sales = 0
        cdef int failed_transactions = 0
        cdef int failed_sales = 0
        cdef int stock = starting_quantity
        cdef list trucks = [0] * self.lead_time

        cdef unsigned int state = np.random.randint(1<<32)

        # Everything else defined later
        cdef int day, customer, request, short, orders

        for day in range(365):
            # A truck arrived
            stock += trucks[day % self.lead_time]
            # This many customers arrive
            for customer in range(random_select(&state, self.itemwise_traffic_zipf_precomp)):
                # This customer wants this many
                request = random_select(&state, self.job_lot_zipf_precomp)
                if stock >= request:
                    # There are enough.
                    successful_transactions += 1
                    successful_sales += request
                    stock -= request
                else:
                    # There are not enough
                    failed_transactions += 1
                    failed_sales += request
            # The day is over. Start making orders.
            if stock < self.safety_stock:
                short = max(self.safety_stock - stock, 0)
                orders = (short + self.order_quantity - 1) // self.order_quantity
                trucks[(day-1) % self.lead_time] = orders * self.order_quantity
        return (
            successful_transactions, successful_sales,
            failed_transactions, failed_sales,
            successful_transactions / (successful_transactions + failed_transactions),
            successful_sales / (successful_sales + failed_sales)
        )
    
    cpdef repeat_simulate_demand(self, int starting_quantity, int count=10000):
        cdef int st = 0
        cdef int ss = 0
        cdef int ft = 0
        cdef int fs = 0
        for _ in range(count):
            xst, xss, xft, xfs, _, _ = self.simulate_demand(starting_quantity)
            st += xst
            ss += xss
            ft += xft
            fs += xfs
        return (st, ss, ft, fs, st/(st+ft), ss/(ss+fs))

# Completely by-the-book reference implementation of xorshift
cdef unsigned int xorshift32(unsigned int* state):
    # Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
    cdef unsigned int x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

# Select an item at random from a buffer
cdef unsigned int random_select(unsigned int* state, np.ndarray[np.int64_t, ndim=1] precomp):
    return precomp[xorshift32(state) % precomp.shape[0]]

if __name__ == "__main__":
    print(Simulation(2, 3, 10, job_lot_zipf=2.0).repeat_simulate_demand(10))
