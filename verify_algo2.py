import numpy as np
import math
import time
import sys
from algorithm_2 import Algorithm2_Sequential

def simulate(model, policy, periods=400000, fixed_s=None):
        state = [0, 0, 0, 0]
        total_profit = 0.0
        waste_a, waste_b = 0, 0
        total_ordered = 0
        
        np.random.seed(42)
        
        for t in range(periods + 1000):
            ia1, ia2, ib1, ib2 = state
            
            if fixed_s:
                curr_a = ia1 + ia2
                curr_b = ib1 + ib2
                qa = max(0, fixed_s[0] - curr_a)
                qb = max(0, fixed_s[1] - curr_b)
            else:
                idx = model._state_to_idx(state)
                qa, qb = policy[idx]

            da = np.random.poisson(model.mu)
            db = np.random.poisson(model.mu)
            
            total_a = ia1 + ia2
            total_b = ib1 + ib2
            
            sa_p = min(da, total_a)
            sb_p = min(db, total_b)
            
            sub_a = min(int(max(0, db - sb_p) * model.gamma), total_a - sa_p)
            sub_b = min(int(max(0, da - sa_p) * model.gamma), total_b - sb_p)
            
            act_sa = sa_p + sub_a
            act_sb = sb_p + sub_b

            sold_a2 = min(act_sa, ia2)
            cur_waste_a = ia2 - sold_a2
            sold_a1 = min(act_sa - sold_a2, ia1)
            rem_a1 = ia1 - sold_a1

            sold_b2 = min(act_sb, ib2)
            cur_waste_b = ib2 - sold_b2
            sold_b1 = min(act_sb - sold_b2, ib1)
            rem_b1 = ib1 - sold_b1
            
            if t >= 1000:
                total_profit += (model.s * (act_sa + act_sb)) - (model.c * (qa + qb))
                waste_a += cur_waste_a
                waste_b += cur_waste_b
                total_ordered += (qa + qb)
            
            state = [qa, rem_a1, qb, rem_b1]

        avg_profit = total_profit / periods
        wa_pct = (waste_a / (total_ordered/2)) * 100
        wb_pct = (waste_b / (total_ordered/2)) * 100
        
        return avg_profit, wa_pct, wb_pct

if __name__ == "__main__":
    print("="*60 + "\nSCENARIO M=2: Two Products, mu=5, s=1, c=.5, gamma=.5\n" + "="*60)
    
    model = Algorithm2_Sequential(M=2, max_inv=10, s=1.0, c=0.5, mu=5.0, gamma=0.5)
    
    print(f"N States: {model.n_states} [Paper: 14641]")
    
    # 1. Verify Base Case (Sa=13, Sb=12)
    prof_base, wa_base, wb_base = simulate(model, None, fixed_s=[13, 12])
    
    # 2. Verify Algorithm 2 (Value Iteration)
    start_time = time.time()
    V, iters, policy = model.value_iteration()
    rt = time.time() - start_time
    
    # 3. Verify Optimal Policy
    prof_opt, wa_opt, wb_opt = simulate(model, policy)
    
    print(f"Convergence: {iters} iterations [Paper expects: 12]")
    print(f"Runtime: {rt:.2f}s")
    print(f"\nBase Stock Sa=13, Sb=12 (PI): {prof_base:.3f} [Paper: 4.479]")
    print(f"Base Stock Waste A: {wa_base:.2f}% [Paper: 6.26%]")
    print(f"Base Stock Waste B: {wb_base:.2f}% [Paper: 5.23%]")
    print(f"\nOptimal Policy (PI): {prof_opt:.3f} [Paper: 5.509]")
    print(f"Optimal Waste A: {wa_opt:.2f}% [Paper: 5.97%]")
    print(f"Optimal Waste B: {wb_opt:.2f}% [Paper: 4.14%]")