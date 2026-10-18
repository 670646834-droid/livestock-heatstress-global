import matplotlib
print("matplotlib", matplotlib.__version__)
try:
    import cartopy
    print("cartopy", cartopy.__version__)
except ImportError:
    print("cartopy MISSING")
try:
    import numpy as np
    print("numpy", np.__version__)
except:
    pass
