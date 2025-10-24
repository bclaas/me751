import importlib

__all__ = [
    # classes
    "Assembly", "RigidBody", "Orientation", "KCon", "DP1", "DP2", "D", "CD",
    # functions
    "vec2quat", "tilde", "A_to_p",
]

_exports = {
    # classes
    "Assembly":    ("simEngine3D.Assembly",    "Assembly"),
    "RigidBody":   ("simEngine3D.Bodies",      "RigidBody"),
    "Orientation": ("simEngine3D.Orientation", "Orientation"),
    "KCon":        ("simEngine3D.KCons",       "KCon"),
    "DP1":         ("simEngine3D.KCons",       "DP1"),
    "DP2":         ("simEngine3D.KCons",       "DP2"),
    "D":           ("simEngine3D.KCons",       "D"),
    "CD":          ("simEngine3D.KCons",       "CD"),
    # functions
    "vec2quat":    ("simEngine3D.Orientation", "vec2quat"),
    "tilde":       ("simEngine3D.Orientation", "tilde"),
    "A_to_p":      ("simEngine3D.Orientation", "A_to_p"),
}

def __getattr__(name):
    try:
        mod_name, attr = _exports[name]
    except KeyError:
        raise AttributeError(f"module 'simEngine3D' has no attribute {name!r}") from None
    obj = getattr(importlib.import_module(mod_name), attr)
    globals()[name] = obj  # cache for next access
    return obj

def __dir__():
    return sorted(list(globals().keys()) + list(__all__))