"""Compatibility fixes for the lox/JAX versions pinned by this project."""

from __future__ import annotations


def patch_lox_scan_metadata() -> None:
    """Teach lox 0.3.1 about JAX 0.11's explicit scan output FlatTree.

    lox appends extracted log arrays to a scan body's jaxpr.  Starting with
    JAX 0.11, the scan primitive also carries ``ft_out`` metadata whose leaf
    count must be extended in lockstep.  Unpatched lox leaves the old count
    in place and every logged PPO/StreamAC/StreamEprop run fails in
    ``_scan_abstract_eval`` before its first update.
    """
    import lox.spooling as spooling

    if getattr(spooling, "_stream_rl_jax011_patch", False):
        return

    try:
        from jax._src.flattree import FTSingleton, FTTuple
    except ImportError:
        return

    original_spool_jaxpr = spooling.spool_jaxpr

    def repair_scan_flat_trees(jaxpr):
        repaired_eqns = []
        changed = False
        for eqn in jaxpr.eqns:
            if eqn.primitive.name != "scan" or "ft_out" not in eqn.params:
                repaired_eqns.append(eqn)
                continue

            inner = eqn.params["jaxpr"]
            inner_jaxpr = inner.jaxpr if hasattr(inner, "jaxpr") else inner
            ft_out = eqn.params["ft_out"]
            missing = len(inner_jaxpr.out_avals) - len(ft_out)
            if missing <= 0:
                repaired_eqns.append(eqn)
                continue

            carry_ft, ys_ft = ft_out.elts
            if not isinstance(ys_ft, FTTuple):
                ys_ft = FTTuple(ys_ft)
            extended_ys_ft = FTTuple(
                *ys_ft.elts, *(FTSingleton(None) for _ in range(missing))
            )
            new_params = {**eqn.params, "ft_out": FTTuple(carry_ft, extended_ys_ft)}
            repaired_eqns.append(eqn.replace(params=new_params))
            changed = True

        return jaxpr.replace(eqns=repaired_eqns) if changed else jaxpr

    def compatible_spool_jaxpr(*args, **kwargs):
        jaxpr, logs = original_spool_jaxpr(*args, **kwargs)
        return repair_scan_flat_trees(jaxpr), logs

    spooling.spool_jaxpr = compatible_spool_jaxpr
    spooling._stream_rl_jax011_patch = True
