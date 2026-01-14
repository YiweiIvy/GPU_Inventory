import torch
from torch import arange, tensor, exp, zeros, clamp
from torch import max as tmax, min as tmin
from torch.distributions import Poisson

class Algorithm1:
    def __init__(self, M, max_inv, s, c, mu, device='cpu'):
        self.M = M
        self.max_inv = max_inv  # Q_bar: the maximum order quantity
        self.s = s
        self.c = c
        self.mu = mu
        self.device = device
        self.n_states = (max_inv + 1) ** M # N = (Q+1)^M
        
        # Pre-compute Poisson demand probabilities
        self.max_demand = int(mu * 10)
        self.demand_range = arange(self.max_demand + 1, device=device)
        mu_t = tensor(mu, device=device)
        # Using Poisson PMF
        self.demand_probs = exp(Poisson(mu_t).log_prob(self.demand_range))
        
        # Pre-compute Esale_j = s * Σ d * pd (capped at total inventory Y)
        self.esales = zeros(self.n_states, device=device)
        for j in range(self.n_states):
            state = self._idx_to_state(j)
            Y = state.sum().item() # Total inventory Y
            d_clipped = clamp(self.demand_range, max=Y)
            self.esales[j] = (d_clipped * self.demand_probs).sum()

    def _idx_to_state(self, idx):
        """Converts vector index j to state (I1, ..., IM)"""
        state = []
        temp = idx
        for _ in range(self.M):
            state.append(temp % (self.max_inv + 1))
            temp //= (self.max_inv + 1)
        return tensor(state, device=self.device)

    def _state_to_idx(self, state):
        """Converts multidimensional state I to vector index j"""
        idx = 0
        for i, val in enumerate(state):
            idx += val.item() * ((self.max_inv + 1) ** i)
        return int(idx)

    def _transition_idx(self, state, q, d):
        """Transition function F(q, I, d)"""
        Y = state.sum().item()
        # Shortcut: F(q,I,d) = (0,0,...,q) as soon as d >= Y
        if d >= Y:
            return q * ((self.max_inv + 1) ** (self.M - 1))
        
        # FIFO Issuing
        rem_inv = state.clone()
        rem_d = d
        for i in range(self.M):
            sold = min(rem_inv[i].item(), rem_d)
            rem_inv[i] -= sold
            rem_d -= sold
            if rem_d <= 0: break
            
        new_state = zeros(self.M, dtype=torch.long, device=self.device)
        if self.M > 1: new_state[:-1] = rem_inv[1:] # Aging
        new_state[-1] = q # Received order
        return self._state_to_idx(new_state)

    def value_iteration(self, tol=1e-4):
        """Value Iteration Algorithm 1"""
        V = (self.s * self.esales).clone() # Initialize Vj to s*Esale
        for it in range(100):
            W = V.clone()
            for j in range(self.n_states):
                state_j = self._idx_to_state(j)
                best_q_val = float('-inf')
                for q in range(self.max_inv + 1):
                    # Σ pd * Wk
                    expected_future = 0.0
                    for d in range(len(self.demand_probs)):
                        prob = self.demand_probs[d].item()
                        if prob < 1e-9: continue
                        k = self._transition_idx(state_j, q, d)
                        expected_future += prob * W[k]
                    
                    q_val = expected_future - (self.c * q) # Term: Σ pd*Wk - cq
                    if q_val > best_q_val: best_q_val = q_val
                
                V[j] = (self.s * self.esales[j]) + best_q_val # Vj = s*Esale + max[...]
            
            # span(V, W) check
            diff = V - W
            span = tmax(diff) - tmin(diff)
            if span < tol: return V, it + 1
        return V, 100

    def extract_policy(self, V):
        """Find Q*(I) = argmax [Σ pd * Vk - cq]"""
        policy = zeros(self.n_states, dtype=torch.long)
        for j in range(self.n_states):
            state_j = self._idx_to_state(j)
            best_q_val, best_q = float('-inf'), 0
            for q in range(self.max_inv + 1):
                expected_future = 0.0
                for d in range(len(self.demand_probs)):
                    prob = self.demand_probs[d].item()
                    if prob < 1e-9: continue
                    k = self._transition_idx(state_j, q, d)
                    expected_future += prob * V[k]
                q_val = expected_future - (self.c * q)
                if q_val > best_q_val:
                    best_q_val, best_q = q_val, q
            policy[j] = best_q
        return policy