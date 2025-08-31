import torch
import torch.nn.functional as F

def attention_rollout(attentions, discard_ratio: float = 0.9, head_fusion: str = "mean"):
    """Compute Attention Rollout as in Abnar & Zuidema (2020).
    Args:
        attentions (List[Tensor]): list of attention tensors from each ViT block,
            each with shape (batch, heads, tokens, tokens).
        discard_ratio (float): fraction of lowest attention weights to discard per head.
        head_fusion (str): one of {"mean", "max"} to fuse heads.
    Returns:
        rollout (Tensor): shape (batch, tokens, tokens) cumulative attention.
    """
    assert len(attentions) > 0, "No attention maps provided"
    with torch.no_grad():
        # fuse heads
        if head_fusion == "mean":
            attn_stack = [a.mean(dim=1) for a in attentions]  # (B, T, T)
        elif head_fusion == "max":
            attn_stack = [a.max(dim=1).values for a in attentions]
        else:
            raise ValueError("head_fusion must be 'mean' or 'max'")
        # normalize & prune
        attn_stack_norm = []
        for A in attn_stack:
            B, T, _ = A.shape
            A = A.clone()
            flat = A.view(B, -1)
            k = (flat.shape[1] * discard_ratio)
            if k >= 1:
                kth_vals = torch.kthvalue(flat, k=int(k), dim=1).values.view(B, 1, 1)
                A[A < kth_vals] = 0.0
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
            attn_stack_norm.append(A + torch.eye(T, device=A.device).unsqueeze(0))  # add residual
        # cumulative product
        rollout = attn_stack_norm[0]
        for A in attn_stack_norm[1:]:
            rollout = A @ rollout
        return rollout  # (B, T, T)
