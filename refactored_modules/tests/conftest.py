import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting in tests; must be set before matplotlib is imported
