import sys
sys.path.append("/home/trent/IsaacLab/env_isaaclab/lib/python3.11/site-packages")
try:
    from pxr import Usd
    stage = Usd.Stage.Open("/home/trent/wheeled_leg/wheeled_leg/assets/HopperTrex.usd")
    for prim in stage.Traverse():
        print(prim.GetPath().pathString)
except Exception as e:
    import builtins
    print(e)
