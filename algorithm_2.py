import numpy as np
import math
import time
import sys

class Algorithm2_Sequential:
    def __init__(self, M=2, max_inv=10, s=1.0, c=0.5, mu=5.0, gamma=0.5):
        self.M, self.Q = M, max_inv
        self.s, self.c, self.mu, self.gamma = s, c, mu, gamma
        self.n_states = (max_inv + 1) ** (2 * M)

        self.max_d = int(mu * 5) # Truncate at negligible probability
        self.probs = []
        for d in range(self.max_d + 1):
            p = (math.exp(-mu) * (mu**d)) / math.factorial(d)
            self.probs.append(p)
        
        self.bases = [(self.Q + 1)**i for i in range(2 * M)]
        
        self.esales_cache = np.zeros(self.n_states)
        self._precompute_esales()

    def _idx_to_state(self, idx):
        state = []
        temp = idx
        for _ in range(2 * self.M):
            state.append(temp % (self.Q + 1))
            temp //= (self.Q + 1)
        return state

    def _state_to_idx(self, state):
        idx = 0
        for i, val in enumerate(state):
            idx += val * self.bases[i]
        return idx

    def _precompute_esales(self):
        for j in range(self.n_states):
            state = self._idx_to_state(j)
            
            expected_rev = 0.0
            for da in range(self.max_d + 1):
                for db in range(self.max_d + 1):
                    prob = self.probs[da] * self.probs[db]
                    if prob < 1e-9: continue
                    
                    rev, _ = self._calc_revenue_and_next_state(state, 0, 0, da, db, only_revenue=True)
                    expected_rev += prob * rev
            
            self.esales_cache[j] = expected_rev

    def _calc_revenue_and_next_state(self, state, qa, qb, da, db, only_revenue=False):
        ia = state[:self.M]
        ib = state[self.M:]
        inv_a = sum(ia)
        inv_b = sum(ib)

        sa_p = min(da, inv_a)
        sb_p = min(db, inv_b)


        unmet_a = max(0, da - sa_p)
        sub_b = min(int(unmet_a * self.gamma), inv_b - sb_p)


        unmet_b = max(0, db - sb_p)
        sub_a = min(int(unmet_b * self.gamma), inv_a - sa_p)

        total_sa = sa_p + sub_a
        total_sb = sb_p + sub_b
        
        revenue = self.s * (total_sa + total_sb)
        
        if only_revenue:
            return revenue, None

        rem_ia = list(ia) # Copy
        sales_rem = total_sa
        taken = min(rem_ia[1], sales_rem)
        rem_ia[1] -= taken
        sales_rem -= taken
        taken = min(rem_ia[0], sales_rem)
        rem_ia[0] -= taken
        
        rem_ib = list(ib)
        sales_rem = total_sb
        taken = min(rem_ib[1], sales_rem)
        rem_ib[1] -= taken
        sales_rem -= taken
        taken = min(rem_ib[0], sales_rem)
        rem_ib[0] -= taken

        
        new_state = [qa, rem_ia[0], qb, rem_ib[0]]
        
        return revenue, self._state_to_idx(new_state)

    def value_iteration(self, tol=1e-4):
        V = np.copy(self.esales_cache)
        policy = np.zeros((self.n_states, 2), dtype=int)
        
        for it in range(100):
            W = np.copy(V)
            start_time = time.time()
            
            for j in range(self.n_states):
                current_state = self._idx_to_state(j)
                best_val = -float('inf')
                best_qa, best_qb = 0, 0
                
                for qa in range(self.Q + 1):
                    for qb in range(self.Q + 1):
                        
                        expected_future_val = 0.0
                        
                        for da in range(self.max_d + 1):
                            for db in range(self.max_d + 1):
                                prob = self.probs[da] * self.probs[db]
                                if prob < 1e-7: continue
                                
                                _, k = self._calc_revenue_and_next_state(current_state, qa, qb, da, db)
                                expected_future_val += prob * W[k]
                        
                        val = expected_future_val - (self.c * (qa + qb))
                        
                        if val > best_val:
                            best_val = val
                            best_qa, best_qb = qa, qb
                
                V[j] = self.esales_cache[j] + best_val
                policy[j] = [best_qa, best_qb]
                
                if j % 100 == 0:
                     sys.stdout.write(f"Iter {it} | State {j}/{self.n_states} | Current Avg V: {np.mean(V[:j+1]):.3f}")
                     sys.stdout.flush()

            span = np.max(V - W) - np.min(V - W)
            
            if span < tol:
                print(f"Converged in {it+1} iterations.")
                return V, it + 1, policy
                
        return V, 100, policy