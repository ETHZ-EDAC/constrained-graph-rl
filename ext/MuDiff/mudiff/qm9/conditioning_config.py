from __future__ import annotations

from typing import List

from qm9.planar_metrics_adapter import canonicalize_metric_list, load_conditioning_metrics_from_use_case


def _merge_unique(first: List[str], second: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()

    for key in list(first) + list(second):
        if key in seen:
            continue
        seen.add(key)
        merged.append(key)

    return merged


def resolve_conditioning_arguments(args):
    raw_cli_metrics = list(getattr(args, "conditioning", []) or [])
    is_planar = "planar" in str(getattr(args, "dataset", "")).lower()

    cli_metrics = canonicalize_metric_list(raw_cli_metrics) if is_planar else raw_cli_metrics

    use_case = getattr(args, "conditioning_use_case", None)
    use_case_dir = getattr(args, "conditioning_use_case_dir", None)
    merge_cli = bool(getattr(args, "conditioning_merge_cli", True))

    if use_case:
        use_case_metrics, use_case_path = load_conditioning_metrics_from_use_case(
            use_case=use_case,
            use_case_dir=use_case_dir,
        )
        if merge_cli:
            resolved = _merge_unique(use_case_metrics, cli_metrics)
        else:
            resolved = use_case_metrics

        args.conditioning = resolved
        args.conditioning_source = f"use_case:{use_case_path}"
        print(f"Loaded conditioning metrics from use-case: {use_case_path}")
    else:
        args.conditioning = cli_metrics
        args.conditioning_source = "cli"

    if is_planar:
        args.conditioning = canonicalize_metric_list(args.conditioning)

    return args
