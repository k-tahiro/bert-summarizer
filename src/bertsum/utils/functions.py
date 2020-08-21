from typing import Any, Callable, Dict


def reduce_dict(f: Callable, a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    return {
        k: f(a[k], b[k])
        for k in a
    }
