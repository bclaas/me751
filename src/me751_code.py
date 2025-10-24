import importlib

__all__ = ["Assembly", "RigidBody", "Orientation", "KCon", "DP1", "DP2", "D", "CD"]

_exports = {
    "Assembly":    ("me751.Assembly",    "Assembly"),
    "RigidBody":   ("me751.Bodies",      "RigidBody"),
    "Orientation": ("me751.Orientation", "Orientation"),
    "KCon":        ("me751.KCons",       "KCon"),
    "DP1":         ("me751.KCons",       "DP1"),
    "DP2":         ("me751.KCons",       "DP2"),
    "D":           ("me751.KCons",       "D"),
    "CD":          ("me751.KCons",       "CD"),
}

def __getattr__(name):
    try:
        mod_name, attr = _exports[name]
    except KeyError:
        raise AttributeError(name) from None
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    globals()[name] = obj
    return obj

def __dir__():
    return sorted(list(globals().keys()) + list(__all__))