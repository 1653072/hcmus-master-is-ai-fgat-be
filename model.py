import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class HFGATDetailed(nn.Module):
    """
    Hierarchical Fashion Graph Attention Network (H-FGAT).
    Encodes items via multimodal features (image + text + category),
    then propagates through outfit and user levels via sparse graph aggregation.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        cat_dim: int,
        num_users: int,
        num_outfits: int,
        num_items: int,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.cat_proj = nn.Linear(cat_dim, embed_dim)

        self.item_fuse = MLPBlock(embed_dim * 3, embed_dim, dropout)
        self.item_update = MLPBlock(embed_dim, embed_dim, dropout)
        self.outfit_update = MLPBlock(embed_dim, embed_dim, dropout)
        self.user_update = MLPBlock(embed_dim, embed_dim, dropout)

        self.user_base = nn.Embedding(num_users, embed_dim)
        self.outfit_base = nn.Embedding(num_outfits, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Name must match the trained checkpoint
        self.compat_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def encode_items(self, image_x, text_x, cat_x, A_item_item):
        xi = F.normalize(self.image_proj(image_x), p=2, dim=-1)
        xt = F.normalize(self.text_proj(text_x), p=2, dim=-1)
        xc = F.normalize(self.cat_proj(cat_x), p=2, dim=-1)

        x = torch.cat([xi, xt, xc], dim=-1)
        x = self.item_fuse(x)
        x = F.normalize(x, p=2, dim=-1)

        x_prop = torch.sparse.mm(A_item_item, x)
        x = x + self.dropout(self.item_update(x_prop))
        return F.normalize(x, p=2, dim=-1)

    def encode_outfits(self, item_emb, A_outfit_item):
        agg = torch.sparse.mm(A_outfit_item, item_emb)
        base = F.normalize(self.outfit_base.weight, p=2, dim=-1)
        out = base + self.dropout(self.outfit_update(agg))
        return F.normalize(out, p=2, dim=-1)

    def encode_users(self, outfit_emb, A_user_outfit):
        agg = torch.sparse.mm(A_user_outfit, outfit_emb)
        base = F.normalize(self.user_base.weight, p=2, dim=-1)
        usr = base + self.dropout(self.user_update(agg))
        return F.normalize(usr, p=2, dim=-1)

    def forward(self, image_x, text_x, cat_x, A_item_item, A_outfit_item, A_user_outfit):
        item_emb = self.encode_items(image_x, text_x, cat_x, A_item_item)
        outfit_emb = self.encode_outfits(item_emb, A_outfit_item)
        user_emb = self.encode_users(outfit_emb, A_user_outfit)
        return user_emb, outfit_emb, item_emb

    def score_compatibility(self, item_emb, outfit_item_batch):
        """
        Score compatibility of outfit(s).

        Args:
            item_emb: Tensor [N_items, embed_dim] — pre-computed item embeddings.
            outfit_item_batch: LongTensor [B, max_items] — item indices per outfit,
                               padded with -1 for missing slots.
        Returns:
            Tensor [B] of compatibility logit scores.
        """
        mask = (outfit_item_batch >= 0).float().unsqueeze(-1)
        safe_idx = outfit_item_batch.clamp_min(0)
        x = item_emb[safe_idx] * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return self.compat_mlp(pooled).squeeze(-1)
