import torch
import torch.nn as nn
import re

class AddChannelEmbedding(nn.Module):
    """
    Adds learned (variable, level) embeddings derived from actual channel names.
    Example channel names:
        "q1000.0", "u850.0", "t500.0", "VAR_2TSFC", "SPSFC", ...

    The module extracts:
        - variable group (prefix)
        - level identifier (pressure like 850.0, or 'SFC')
    Builds embeddings and adds them to input x of shape (B, C, H, W).
    """

    def __init__(self, channel_names, feature_dim, embed_dim=8):
        """
        Args:
            channel_names (list of str): Names for each input channel.
            feature_dim (int): Output feature dimension per channel (projection size).
            embed_dim (int): Embedding dimension for variable and level embeddings.
        """
        super().__init__()

        self.channel_names = channel_names
        C = len(channel_names)

        # -----------------------------
        # PARSE VARIABLE + LEVEL NAMES
        # -----------------------------
        variable_ids = []
        level_ids = []

        for name in channel_names:
            var, lev = self._parse_name(name)
            variable_ids.append(var)
            level_ids.append(lev)

        # Get unique sorted sets
        unique_vars = sorted(list(set(variable_ids)))
        unique_levs = sorted(list(set(level_ids)), key=lambda x: (x != 'SFC', float(x) if x != 'SFC' else -9999))

        # Map string → integer ID
        self.var_to_id = {v: i for i, v in enumerate(unique_vars)}
        self.lev_to_id = {l: i for i, l in enumerate(unique_levs)}

        # Store channel → (var_id, lev_id)
        mapping = []
        for var, lev in zip(variable_ids, level_ids):
            mapping.append([self.var_to_id[var], self.lev_to_id[lev]])
        self.register_buffer("channel_map", torch.tensor(mapping, dtype=torch.long))  # (C, 2)

        # -----------------------------
        # LEARNED EMBEDDINGS
        # -----------------------------
        self.var_emb = nn.Parameter(torch.randn(len(unique_vars), embed_dim))
        self.lev_emb = nn.Parameter(torch.randn(len(unique_levs), embed_dim))

        # Project combined embedding to desired feature dimension
        self.channel_proj = nn.Linear(embed_dim, 1)

    # ----------------------------------------------------------------------
    # NAME PARSING LOGIC
    # ----------------------------------------------------------------------
    def _parse_name(self, name):
        """
        Extracts:
            variable: e.g. 'q', 'u', 'v', 't', 'VAR_2T', 'SP', etc.
            level: pressure (string like '850.0') or 'SFC'
        Rules:
            - If name ends in 'SFC' → level = 'SFC', variable = prefix
            - Else: extract number at the end as pressure
        """

        # Surface case
        if name.endswith("SFC"):
            lev = "SFC"
            var = name[:-3]   # remove 'SFC'
            return var, lev

        # Pressure level case: extract trailing number
        m = re.search(r"(\d+(\.\d+)?)$", name)
        if m:
            lev = m.group(1)
            var = name[:m.start()]  # everything before the number
            return var, lev

        # Fallback: treat entire name as variable, with level = 'UNK'
        return name, "UNK"

    # ----------------------------------------------------------------------
    # FORWARD: ADD EMBEDDINGS
    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x: (B, C, H, W)
        Output: x + embedding
        """
        B, C, H, W = x.shape
        assert C == self.channel_map.shape[0], "Mismatch: input channels vs names list"

        var_ids = self.channel_map[:, 0]
        lev_ids = self.channel_map[:, 1]

        # lookup embeddings
        var_vecs = self.var_emb[var_ids]      # (C, embed_dim)
        lev_vecs = self.lev_emb[lev_ids]      # (C, embed_dim)

        
        # combine
        ch_emb = var_vecs + lev_vecs          # (C, embed_dim)

        # project
        ch_emb = self.channel_proj(ch_emb)    # (C, feature_dim)

        # broadcast to spatial dimensions
        ch_emb = ch_emb.view(1, C, 1, 1)
        
        return x + ch_emb
