"""Fareground env kernel CLI.

Run with:

    python -m fg_env_kernel <command> [args...]

Or from the project root with the right PYTHONPATH:

    python -m kernel <command> [args...]

## Commands

  compile <file.json> [--smoke ROUNDS] [--seed N] [--no-lint]
      Validate + lint a JSON template. With --smoke, also run a fast
      mocked-decision playtest. Exit code 0 = clean, 1 = compile failed,
      2 = compile clean but smoke failed.

  contract [-o FILE]
      Dump the kernel JSON Schema + live capabilities. Stdout by default.

  capabilities
      Print just the live capabilities (effect ops, terminations, etc.).

  versions
      List all known contract versions in the migration chain.

  lint <file.json>
      Run lint only (no engine build). Useful in pre-commit hooks.

## Examples

    python -m kernel compile assets/tic_tac_toe/template.json --smoke 20
    python -m kernel contract -o docs/kernel_contract.json
    python -m kernel capabilities
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def cmd_compile(args: argparse.Namespace) -> int:
    from . import compile_template, smoke_test

    try:
        raw = _load_json(args.file)
    except Exception as e:
        print(f"error reading {args.file}: {e}", file=sys.stderr)
        return 1

    result = compile_template(
        raw,
        seed=args.seed,
        skip_lint=args.no_lint,
    )

    if result.warnings:
        print(f"\n{len(result.warnings)} warning(s):", file=sys.stderr)
        for w in result.warnings:
            print(f"  {w}", file=sys.stderr)

    if not result.ok:
        print(f"\ncompile FAILED ({len(result.errors)} error(s))", file=sys.stderr)
        for e in result.errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print(f"compile OK — {len(result.state.entities)} entities, "
          f"{len(result.state.action_definitions)} actions, "
          f"{len(result.engine.termination_conditions)} termination(s)")

    if args.smoke:
        print(f"\nsmoke_test (rounds={args.smoke}, seed={args.seed}):")
        report = smoke_test(result.engine, rounds=args.smoke, seed=args.seed)
        print(report.summary())
        if not report.healthy:
            return 2

    return 0


def cmd_contract(args: argparse.Namespace) -> int:
    from . import export_kernel_contract

    contract = export_kernel_contract()
    data = json.dumps(contract, indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(data)
        print(f"wrote {args.output} ({len(data):,} bytes)")
    else:
        print(data)
    return 0


def cmd_capabilities(args: argparse.Namespace) -> int:
    from . import export_kernel_contract

    contract = export_kernel_contract()
    caps = contract.get("live_capabilities", {})
    print(f"Kernel capabilities (contract v{contract.get('version', '?')}):")
    for category in sorted(caps.keys()):
        items = caps[category]
        print(f"\n  {category}:")
        for item in sorted(items):
            print(f"    - {item}")
    return 0


def cmd_versions(args: argparse.Namespace) -> int:
    from .pipeline.versioning import CONTRACT_VERSION, list_known_versions

    versions = list_known_versions()
    print(f"current: {CONTRACT_VERSION}")
    print("all known versions (in migration order):")
    for v in versions:
        marker = " ← current" if v == CONTRACT_VERSION else ""
        print(f"  - {v}{marker}")
    return 0


def cmd_primitives(args: argparse.Namespace) -> int:
    """List every primitive registered in the live kernel."""
    from . import list_loaded_primitives
    # Force a re-scan in case files were added since the package was
    # first imported (CLI tools may be run after scaffolding).
    from .primitives_loader import discover as _discover
    _discover()

    loaded = list_loaded_primitives()
    if args.kind:
        items = loaded.get(args.kind, [])
        for item in items:
            print(item)
        return 0
    for category in sorted(loaded.keys()):
        items = loaded[category]
        print(f"\n## {category} ({len(items)})")
        for item in items:
            print(f"  - {item}")
    return 0


def cmd_new_primitive(args: argparse.Namespace) -> int:
    """Scaffold a new primitive file from a template."""
    from .primitives_loader import PRIMITIVE_TEMPLATES, _repo_primitives_dir

    if args.kind not in PRIMITIVE_TEMPLATES:
        print(f"unknown kind '{args.kind}'. Available: "
              f"{', '.join(sorted(PRIMITIVE_TEMPLATES.keys()))}", file=sys.stderr)
        return 1

    template = PRIMITIVE_TEMPLATES[args.kind]
    # Resolve destination directory
    if args.output:
        target_dir = Path(args.output).expanduser().resolve()
    else:
        target_dir = _repo_primitives_dir()
        if target_dir is None:
            print("no kernel_primitives/ directory found; pass --output PATH",
                  file=sys.stderr)
            return 1
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"{args.name}.py"
    if file_path.exists() and not args.force:
        print(f"{file_path} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    description = args.description or f"Custom {args.kind} primitive."
    class_name = "".join(part.capitalize() for part in args.name.split("_")) + "Resolution"
    content = template.format(
        name=args.name,
        description=description,
        ClassName=class_name,
    )
    file_path.write_text(content)
    print(f"scaffolded {file_path}")
    print("\nNext: edit the file, then run `python -m kernel primitives` to verify it loaded.")
    return 0


def cmd_scaffold_env(args: argparse.Namespace) -> int:
    """Scaffold a fresh env package directory."""
    from . import scaffold_env

    target = Path(args.directory).expanduser().resolve()
    if target.exists() and not args.force:
        print(f"{target} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    pkg = scaffold_env(target, args.name)
    print(f"scaffolded env '{args.name}' at {target}")
    print("  files: meta.json, overview.md, template.json")
    print(f"  next: edit, then `python -m kernel compile {target}/template.json`")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    """Zip an env directory into a .simworld archive."""
    from . import pack_env

    out = args.output
    archive = pack_env(args.directory, archive_path=out)
    print(f"packed → {archive} ({archive.stat().st_size:,} bytes)")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    """Extract a .simworld archive into a directory."""
    from . import unpack_env

    target = args.directory
    out = unpack_env(args.archive, directory=target)
    print(f"unpacked → {out}")
    return 0


def cmd_env_info(args: argparse.Namespace) -> int:
    """Show meta + validation for an env package."""
    from . import load_env_package, validate_package

    try:
        pkg = load_env_package(args.path, compile=not args.no_compile)
    except Exception as e:
        print(f"failed to load: {e}", file=sys.stderr)
        return 1

    print(f"env: {pkg.name}")
    print(f"  source: {pkg.source_path}")
    if pkg.meta:
        for key in ("display_name", "version", "contract_version", "author",
                    "created", "updated"):
            if key in pkg.meta:
                print(f"  {key}: {pkg.meta[key]}")
        if pkg.meta.get("tags"):
            print(f"  tags: {', '.join(pkg.meta['tags'])}")
    print(f"  overview: {len(pkg.overview)} chars")
    print(f"  viz files: {len(pkg.viz_files)}")
    print(f"  primitive files: {len(pkg.primitive_files)}")

    issues = validate_package(pkg)
    if issues:
        print("\nPackage issues:")
        for i in issues:
            print(f"  ! {i}")

    if pkg.compile_result is not None:
        if pkg.compile_result.ok:
            print("\ncompile: OK")
            if pkg.compile_result.warnings:
                print(f"  ({len(pkg.compile_result.warnings)} warnings)")
        else:
            print(f"\ncompile: FAILED ({len(pkg.compile_result.errors)} errors)")
            for e in pkg.compile_result.errors:
                print(f"  {e}")
            return 1
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from . import replay

    try:
        raw = _load_json(args.file)
    except Exception as e:
        print(f"error reading {args.file}: {e}", file=sys.stderr)
        return 1

    trace = replay(
        raw,
        seed=args.seed,
        rounds=args.rounds,
        decisions=args.decisions,
    )

    if args.json:
        print(trace.to_json())
    else:
        print(trace.summary())

    if args.output:
        Path(args.output).write_text(trace.to_json())
        print(f"\nwrote trace to {args.output}", file=sys.stderr)

    return 0 if trace.completed and not trace.crash else 2


def cmd_lint(args: argparse.Namespace) -> int:
    from . import lint_template

    try:
        raw = _load_json(args.file)
    except Exception as e:
        print(f"error reading {args.file}: {e}", file=sys.stderr)
        return 1

    issues = lint_template(raw)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    for w in warnings:
        print(f"[warning] {w.path}: {w.message}")
        if w.hint:
            print(f"  hint: {w.hint}")
    for e in errors:
        print(f"[error] {e.path}: {e.message}", file=sys.stderr)
        if e.hint:
            print(f"  hint: {e.hint}", file=sys.stderr)

    if errors:
        print(f"\nlint FAILED ({len(errors)} error, {len(warnings)} warning)", file=sys.stderr)
        return 1
    print(f"lint OK ({len(warnings)} warning)")
    return 0


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kernel",
        description="Fareground env kernel CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compile = sub.add_parser("compile", help="validate, lint, optionally smoke-test a JSON template")
    p_compile.add_argument("file", help="path to template.json")
    p_compile.add_argument("--smoke", type=int, metavar="ROUNDS",
                           help="also run smoke_test for ROUNDS rounds")
    p_compile.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    p_compile.add_argument("--no-lint", action="store_true", help="skip static lint")
    p_compile.set_defaults(func=cmd_compile)

    p_contract = sub.add_parser("contract", help="dump JSON Schema + live capabilities")
    p_contract.add_argument("-o", "--output", help="write to file instead of stdout")
    p_contract.set_defaults(func=cmd_contract)

    p_caps = sub.add_parser("capabilities", help="print live capability lists")
    p_caps.set_defaults(func=cmd_capabilities)

    p_versions = sub.add_parser("versions", help="list contract versions")
    p_versions.set_defaults(func=cmd_versions)

    p_lint = sub.add_parser("lint", help="run lint only (no engine build)")
    p_lint.add_argument("file", help="path to template.json")
    p_lint.set_defaults(func=cmd_lint)

    p_prims = sub.add_parser("primitives", help="list every registered primitive by namespace")
    p_prims.add_argument("--kind", choices=["effects", "preconditions", "resolutions",
                                            "phases", "terminations", "modules",
                                            "target_selectors", "triggers"],
                         help="filter to one namespace")
    p_prims.set_defaults(func=cmd_primitives)

    p_new = sub.add_parser("new-primitive", help="scaffold a new primitive file")
    p_new.add_argument("--kind", required=True,
                        choices=["effect", "termination", "resolution",
                                 "precondition", "expr_func"],
                        help="what kind of primitive to scaffold")
    p_new.add_argument("--name", required=True, help="snake_case identifier")
    p_new.add_argument("--description", help="one-line summary")
    p_new.add_argument("--output", help="target directory (default: <repo>/kernel_primitives/)")
    p_new.add_argument("--force", action="store_true", help="overwrite existing file")
    p_new.set_defaults(func=cmd_new_primitive)

    # ── Env package commands ──────────────────────────────────────
    p_scaffold = sub.add_parser("scaffold-env", help="create a fresh env package directory")
    p_scaffold.add_argument("directory", help="target directory path")
    p_scaffold.add_argument("--name", required=True, help="env slug (snake_case)")
    p_scaffold.add_argument("--force", action="store_true",
                            help="overwrite existing directory")
    p_scaffold.set_defaults(func=cmd_scaffold_env)

    p_pack = sub.add_parser("pack", help="zip an env directory into a .simworld archive")
    p_pack.add_argument("directory", help="env directory to pack")
    p_pack.add_argument("-o", "--output", help="archive path (default: <dir>.simworld)")
    p_pack.set_defaults(func=cmd_pack)

    p_unpack = sub.add_parser("unpack", help="extract a .simworld archive")
    p_unpack.add_argument("archive", help=".simworld file to unpack")
    p_unpack.add_argument("-d", "--directory",
                          help="target directory (default: archive name)")
    p_unpack.set_defaults(func=cmd_unpack)

    p_info = sub.add_parser("env-info", help="show metadata + compile status of an env package")
    p_info.add_argument("path", help="env directory or .simworld archive")
    p_info.add_argument("--no-compile", action="store_true",
                        help="skip compile (faster, but doesn't validate runtime)")
    p_info.set_defaults(func=cmd_env_info)

    p_replay = sub.add_parser("replay", help="deterministic step-by-step trace")
    p_replay.add_argument("file", help="path to template.json")
    p_replay.add_argument("--seed", type=int, default=42)
    p_replay.add_argument("--rounds", type=int, default=50)
    p_replay.add_argument("--decisions", default="random",
                          choices=["random", "first", "round_robin"])
    p_replay.add_argument("--json", action="store_true",
                          help="output full JSON trace instead of summary")
    p_replay.add_argument("-o", "--output", help="write JSON trace to file")
    p_replay.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
