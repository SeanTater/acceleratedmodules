#!/usr/bin/env python3
import numpy as np

class Simulation:
    ''' Simulates the demand of one product over time
    
        Safety stock is how many items to target on the shelf at all times.
        Lead time is how many days between making an order and when it arrives on the shelf.
        Order quantity is how many items you can order at a time (orders are an integer multiple of this)
        Job lot zipf is the zipf parameter of the distribution of job lot sizes - how many people buy at a time
        Itemwise traffic zipf is analogous, but for the distribution of customer traffic for that item
    '''
    def __init__(self, safety_stock, lead_time, order_quantity, job_lot_zipf=2.75, itemwise_traffic_zipf=4.0):
        self.safety_stock = safety_stock
        self.lead_time = lead_time
        self.order_quantity = order_quantity
        self.job_lot_zipf = job_lot_zipf
        self.itemwise_traffic_zipf = itemwise_traffic_zipf

    def simulate_demand(self, starting_quantity):
        successful_transactions = 0
        successful_sales = 0
        failed_transactions = 0
        failed_sales = 0
        stock = starting_quantity
        trucks = [0] * self.lead_time

        for day in range(365):
            # A truck arrived
            stock += trucks[day % self.lead_time]
            # This many customers arrive
            for customer in range(np.random.zipf(self.itemwise_traffic_zipf)):
                # This customer wants this many
                request = np.random.zipf(self.job_lot_zipf)
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
    
    def repeat_simulate_demand(self, starting_quantity, count=10000):
        st, ss, ft, fs = 0, 0, 0, 0
        for _ in range(count):
            xst, xss, xft, xfs, _, _ = self.simulate_demand(starting_quantity)
            st += xst
            ss += xss
            ft += xft
            fs += xfs
        return (st, ss, ft, fs, st/(st+ft), ss/(ss+fs))
    

if __name__ == "__main__":
    print(Simulation(2, 3, 10, job_lot_zipf=2.0).repeat_simulate_demand(10))
