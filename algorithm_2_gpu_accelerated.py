import torch
import time
import numpy as np

class Algorithm2_MPS_GPU_Accelerated:
    def __init__(self, M=2, Q=10, s=1.0, c=0.5, mu=5.0, gamma=0.5):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.M, self.Q = M, Q
        self.s, self.c, self.mu, self.gamma = s, c, mu, gamma
        self.n_states = (Q + 1) ** (2 * M)

        # Pre-compute Joint Poisson
        self.max_d = int(mu * 5)
        d_range = torch.arange(self.max_d + 1, device=self.device).float()
        p = torch.exp(torch.distributions.Poisson(torch.tensor(mu, device=self.device)).log_prob(d_range))
        self.joint_probs = p.unsqueeze(1) * p.unsqueeze(0)

        self.bases = (Q + 1) ** torch.arange(2 * M, device=self.device)
        self.states = self._generate_all_states()
        
        print(f"[*] Pre-computing revenue...")
        self.esales = self._compute_revenue_vectorized()

    def _generate_all_states(self):
        ranges = [torch.arange(self.Q + 1, device=self.device)] * (2 * self.M)
        grid = torch.meshgrid(*ranges, indexing='ij')
        return torch.stack(grid, dim=-1).reshape(-1, 2 * self.M)

    def _compute_revenue_vectorized(self):
        inv_a = self.states[:, :self.M].sum(dim=1).view(-1, 1, 1)
        inv_b = self.states[:, self.M:].sum(dim=1).view(-1, 1, 1)
        d_a = torch.arange(self.max_d + 1, device=self.device).view(1, -1, 1)
        d_b = torch.arange(self.max_d + 1, device=self.device).view(1, 1, -1)
        
        sa_p = torch.minimum(d_a, inv_a)
        sb_p = torch.minimum(d_b, inv_b)
        sub_a = torch.minimum(((d_b - sb_p) * self.gamma).long(), inv_a - sa_p)
        sub_b = torch.minimum(((d_a - sa_p) * self.gamma).long(), inv_b - sb_p)
        
        return (self.joint_probs * (self.s * (sa_p + sub_a + sb_p + sub_b))).sum(dim=(1, 2))

    def get_next_indices(self, qa, qb):
        # Full FIFO and Substitution Logic
        ia = self.states[:, :self.M]
        ib = self.states[:, self.M:]
        inv_a, inv_b = ia.sum(dim=1).view(-1,1,1), ib.sum(dim=1).view(-1,1,1)
        
        da = torch.arange(self.max_d + 1, device=self.device).view(1, -1, 1)
        db = torch.arange(self.max_d + 1, device=self.device).view(1, 1, -1)
        
        sa_p = torch.minimum(da, inv_a)
        sb_p = torch.minimum(db, inv_b)
        # Symmetrical substitution
        total_sa = sa_p + torch.minimum(((db - sb_p) * self.gamma).long(), inv_a - sa_p)
        total_sb = sb_p + torch.minimum(((da - sa_p) * self.gamma).long(), inv_b - sa_p)

        # FIFO for M=2
        # State: (Ia1, Ia2, Ib1, Ib2) -> (Ia2_rem, qa, Ib2_rem, qb)
        rem_ia2 = torch.clamp(ia[:, 1].view(-1,1,1) - torch.clamp(total_sa - ia[:, 0].view(-1,1,1), min=0), min=0)
        rem_ib2 = torch.clamp(ib[:, 1].view(-1,1,1) - torch.clamp(total_sb - ib[:, 0].view(-1,1,1), min=0), min=0)

        return (rem_ia2 * self.bases[0] + qa * self.bases[1] + rem_ib2 * self.bases[2] + qb * self.bases[3]).long()

    def value_iteration(self, tol=1e-4):
        V = self.esales.clone()
        best_policy = torch.zeros((self.n_states, 2), dtype=torch.long, device=self.device)

        for it in range(100):
            W = V.clone()
            best_val = torch.full((self.n_states,), -float('inf'), device=self.device)
            
            for qa in range(self.Q + 1):
                for qb in range(self.Q + 1):
                    k = self.get_next_indices(qa, qb)
                    # Expected future value: Σ p(d) * W(k)
                    ev = (self.joint_probs * W[k]).sum(dim=(1, 2)) - (self.c * (qa + qb))
                    
                    mask = ev > best_val
                    best_val[mask] = ev[mask]
                    if it == 99 or it > 10: # Only capture policy near convergence
                        best_policy[mask, 0] = qa
                        best_policy[mask, 1] = qb

            V = self.esales + best_val
            span = torch.max(V - W) - torch.min(V - W)
            print(f"    Iter {it}: Span = {span.item():.6f}")
            if span < tol: return V, it + 1, best_policy
        return V, 100, best_policy