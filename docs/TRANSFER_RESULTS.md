# OfferLab Transfer Results

Wave 4 treats external benchmark sources as ancillary validation, not as pooled
training rows for eBay offer prediction.

Current default ablation status: **not run**. No numeric hidden-loss or calibration values are fabricated when measured results have not been supplied.

Reason:

- The NBER Best Offer benchmark is the direct behavioral evidence source.
- Open Bandit and Criteo validate evaluator and causal machinery, but their rows
  are not eBay bargaining rows.
- CraigslistBargain is useful for dialogue-act extraction only.
- Transfer is retained only after explicit measured NBER hidden-loss and calibration results both improve beyond preregistered thresholds.

Policy:

- Raw-row pooling across domains is disabled.
- Transfer artifacts are research-only.
- Production export remains blocked unless a future artifact is trained on
  authorized seller data with confirmed commercial-use rights.
