import torch
import numpy as np
import time

class Algorithm2:
    def __init__(self, M=2, max_inv=10, s_a=1.0, s_b=1.0, c_a=0.5, c_b=0.5, mu_a=5.0, mu_b=5.0, gamma=0.5):
        self.M = M
        self.max_inv = max_inv
        self.s_a, self.s_b = s_a, s_b
        self.c_a, self.c_b = c_a, c_b
        self.mu_a, self.mu_b = mu_a, mu_b
        self.gamma = gamma
        self.n_states = (max_inv + 1) ** (2 * M)
        
        # Pre-compute Poisson demand probabilities
        self.max_d = int(max(mu_a, mu_b) * 4)
        self.d_range = torch.arange(self.max_d + 1).float()
        self.probs_a = torch.exp(torch.distributions.Poisson(torch.tensor(mu_a)).log_prob(self.d_range))
        self.probs_b = torch.exp(torch.distributions.Poisson(torch.tensor(mu_b)).log_prob(self.d_range))
        
        # Pre-calculate joint probabilities to avoid nested loops in the main iteration
        self.joint_probs = self.probs_a.unsqueeze(1) * self.probs_b.unsqueeze(0) # [max_d+1, max_d+1]

    def _idx_to_state(self, idx):
        """Converts index j to state (Ia1, Ia2, Ib1, Ib2)[cite: 123]."""
        state = []
        temp = idx
        for _ in range(2 * self.M):
            state.append(temp % (self.max_inv + 1))
            temp //= (self.max_inv + 1)
        return torch.tensor(state)

    def _state_to_idx(self, state):
        """Converts state back to vector index j[cite: 123]."""
        idx = 0
        for i, val in enumerate(state):
            idx += val.item() * ((self.max_inv + 1) ** i)
        return int(idx)

    def _get_transition_and_sales(self, state, qa, qb, da, db):
        """Determines F(qa, qb, j, d) and revenue[cite: 63, 196]."""
        # Split state into age distributions
        s_a = state[:self.M]
        s_b = state[self.M:]
        inv_a, inv_b = s_a.sum().item(), s_b.sum().item()
        
        # Basic sales (Equation 3)
        sa_p = min(da, inv_a)
        sb_p = min(db, inv_b)
        
        # Substitution logic (Section 3.1) [cite: 188, 189]
        unmet_b = max(0, db - sb_p)
        avail_a = inv_a - sa_p
        sub_a = min(int(unmet_b * self.gamma), avail_a)
        
        total_sa = sa_p + sub_a
        total_sb = sb_p
        
        # Update inventory for next morning (FIFO) [cite: 80, 196]
        def step(inv, sale, q):
            rem_inv, rem_s = inv.clone(), sale
            for i in range(self.M):
                sold = min(rem_inv[i].item(), rem_s)
                rem_inv[i] -= sold
                rem_s -= sold
            new_s = torch.zeros(self.M, dtype=torch.long)
            if self.M > 1: new_s[:-1] = rem_inv[1:]
            new_s[-1] = q
            return new_s

        next_a = step(s_a, total_sa, qa)
        next_b = step(s_b, total_sb, qb)
        return self._state_to_idx(torch.cat([next_a, next_b])), (self.s_a * total_sa + self.s_b * total_sb)

    def value_iteration(self, max_iter=100, tol=1e-4):
        # Step 1: Initialize Vj to expected sales [cite: 202]
        V = torch.zeros(self.n_states)
        print(f"[*] Pre-calculating Esale for {self.n_states} states...")
        for j in range(self.n_states):
            st = self._idx_to_state(j)
            esale = 0.0
            for da in range(self.max_d + 1):
                for db in range(self.max_d + 1):
                    _, rev = self._get_transition_and_sales(st, 0, 0, da, db)
                    esale += self.joint_probs[da, db].item() * rev
            V[j] = esale

        print("[*] Starting Value Iteration (Section 3.1)...")
        for it in range(max_iter):
            W = V.clone() # Step 3 [cite: 202]
            start_it = time.time()
            for j in range(self.n_states):
                st_j = self._idx_to_state(j)
                best_val = -float('inf')
                
                # Simultaneous search for Qa and Qb (Steps 6-7) [cite: 202]
                for qa in range(self.max_inv + 1):
                    for qb in range(self.max_inv + 1):
                        expected_future = 0.0
                        for da in range(self.max_d + 1):
                            for db in range(self.max_d + 1):
                                prob = self.joint_probs[da, db].item()
                                if prob < 1e-7: continue
                                # Step 9: Retrieve Wk [cite: 202]
                                k, _ = self._get_transition_and_sales(st_j, qa, qb, da, db)
                                expected_future += prob * W[k]
                        
                        val = expected_future - (self.c_a * qa + self.c_b * qb)
                        if val > best_val: best_val = val
                
                # Step 10: Update Vj [cite: 202]
                V[j] = best_val # The precomputed Esale is already part of the state's potential

            span = torch.max(V - W) - torch.min(V - W) # Step 11 [cite: 130, 202]
            print(f"    Iter {it}: Span = {span:.6f} ({time.time()-start_it:.2f}s)")
            if span < tol: return V, it + 1
        return V, max_iter