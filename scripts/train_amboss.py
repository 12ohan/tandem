#!/usr/bin/env python3
"""CLI runner script forwarding to tandem.train_amboss."""

import sys
from pathlib import Path

# Ensure tandem package is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tandem.train_amboss import main

if __name__ == "__main__":
    main()
