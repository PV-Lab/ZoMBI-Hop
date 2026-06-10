"""
Backward-compatible shim — use synthetic_data/generate_synthetic_campaign.py.

  python synthetic_data/generate_synthetic_campaign.py
  python synthetic_data/generate_synthetic_campaign.py --all-3d
"""

from __future__ import annotations

import sys
import warnings

if __name__ == "__main__":
    warnings.warn(
        "generate_synthetic_campaign_3d.py is a shim; "
        "use synthetic_data/generate_synthetic_campaign.py",
        DeprecationWarning,
        stacklevel=1,
    )
    from synthetic_data.generate_synthetic_campaign import main
    main()
