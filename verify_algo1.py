import torch
import numpy as np
import time
from algorithm_1 import Algorithm1

def simulate(model, policy_type, S=None, V=None, T=400000):
    np.random.seed(42)
    state = torch.zeros(model.M, dtype=torch.long)
    policy = model.extract_policy(V) if policy_type == "optimal" else None
    
    total_profit, total_waste, total_order = 0.0, 0.0, 0.0
    for t in range(T + 1000): # 1000 step warm-up
        if policy_type == "optimal":
            q = policy[model._state_to_idx(state)].item()
        else:
            # Base stock policy: Q = (S - I)+
            q = max(0, S - state.sum().item())
            q = min(q, model.max_inv)

        demand = np.random.poisson(model.mu)
        inv = state.clone()
        sold, rem_d = 0, demand
        for i in range(model.M):
            qty = min(inv[i].item(), rem_d)
            inv[i] -= qty
            sold += qty
            rem_d -= qty
        
        waste = inv[0].item() # Waste calculation (I1 - d)+
        
        if t >= 1000:
            total_profit += (model.s * sold - model.c * q) # Profit formula
            total_waste += waste
            total_order += q
        
        # State transition
        next_s = torch.zeros(model.M, dtype=torch.long)
        if model.M > 1: next_s[:-1] = inv[1:]
        next_s[-1] = q
        state = next_s

    return total_profit/T, (total_waste/total_order)*100

def run_all_verifications():
    # --- M=2 Scenario ---
    print("="*60 + "\nSCENARIO M=2: mu=5, s=1, c=.5\n" + "="*60)
    m2 = Algorithm1(M=2, max_inv=9, s=1.0, c=0.5, mu=5.0)
    v2, it2 = m2.value_iteration()
    p2_opt, w2_opt = simulate(m2, "optimal", V=v2)
    p2_base, w2_base = simulate(m2, "base_stock", S=13)
    
    print(f"Convergence: {it2} iterations [Paper expects: 12] ")
    print(f"Optimal Policy (pi): {p2_opt:.3f} [Paper: 2.215]")
    print(f"Optimal Waste: {w2_opt:.2f}% [Paper: 5.78%]")
    print(f"Base Stock S=13 (PI): {p2_base:.3f} [Paper: 2.195]")
    print(f"Base Stock Waste: {w2_base:.2f}% [Paper: 7.33%]")

    # --- M=3 Scenario --- 
    print("\n" + "="*60 + "\nSCENARIO M=3: mu=5, s=1, c=.5\n" + "="*60)
    m3 = Algorithm1(M=3, max_inv=15, s=1.0, c=0.5, mu=5.0)
    v3, it3 = m3.value_iteration()
    p3_opt, w3_opt = simulate(m3, "optimal", V=v3)
    p3_base, w3_base = simulate(m3, "base_stock", S=15)
    
    print(f"N States: {m3.n_states} [Paper: 4096] ")
    print(f"Convergence: {it3} iterations [Paper: 15] ")
    print(f"Optimal Policy (pi): {p3_opt:.3f} [Paper: 2.40] ")
    print(f"Optimal Waste: {w3_opt:.2f}% [Paper: 2.53%] ")
    print(f"Base Stock S=15 (PI): {p3_base:.3f} [Paper: 2.39] ")
    print(f"Base Stock Waste: {w3_base:.2f}% [Paper: 2.63%] ")

    # --- M=4 Scenario ---
    print("\n" + "="*60 + "\nSCENARIO M=4: mu=5, s=1, c=.5\n" + "="*60)
    m4 = Algorithm1(M=4, max_inv=20, s=1.0, c=0.5, mu=5.0)
    # The paper notes pi=2.47 matches a base stock of S=16 
    p4_base, w4_base = simulate(m4, "base_stock", S=16)
    print(f"Base Stock S=16 (PI): {p4_base:.3f} [Paper Value: 2.47] ")

if __name__ == "__main__":
    run_all_verifications()