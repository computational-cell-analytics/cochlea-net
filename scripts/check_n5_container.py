"""Check an n5 or zarr container for an incomplete or corrupted copy.

A dataset directory whose 'attributes.json' is missing, empty or without the 'dimensions'
key reads as a *group*, not as an array. Every reader then fails far away from the cause,
for example with "AttributeError: 'Group' object has no attribute 'shape'" inside the
prediction. This script names the cause instead.

The structural checks use the standard library only, so the script also runs in an
environment where z5py, zarr or elf are missing or broken - which is one of the cases it
diagnoses.

Usage:
    python check_n5_container.py PATH [-k setup0/timepoint0/s0]
"""
import argparse
import importlib
import itertools
import json
import os
import sys
from math import ceil, prod

# The minimum size of a valid n5 chunk file: mode (2 bytes) + number of dimensions (2 bytes).
# Same criterion as 'find_truncated_chunks' in scripts/data_transfer/smb_transfer_resilient.py.
MIN_CHUNK_BYTES = 4

N5_ATTRS = "attributes.json"
ZARR2_ARRAY, ZARR2_GROUP, ZARR3_NODE = ".zarray", ".zgroup", "zarr.json"

DATASET, GROUP, ZARR_ARRAY, ZARR_GROUP, BROKEN = (
    "n5 dataset", "n5 group", "zarr array", "zarr group", "corrupt metadata"
)
# A group needs no 'attributes.json'. Only a directory that holds chunk data without metadata,
# or one that holds nothing at all, is a defect.
IMPLICIT_GROUP, NO_METADATA, EMPTY_DIR = (
    "n5 group (implicit)", "dataset without metadata", "empty directory"
)


def _normalize_key(key):
    """Normalize a container key. Keys use '/' on every platform, so accept either
    separator. The root node is keyed as '/'."""
    parts = [p for p in key.replace("\\", "/").split("/") if p not in ("", ".")]
    return "/".join(parts) or "/"


def _node_dir(container, key):
    """Return the directory of a node. The key is '/'-separated, the path is not."""
    return container if key == "/" else os.path.join(container, *key.split("/"))


def _container_extension(path):
    """Return the lowercase extension. The reader is picked by it, so it must be exact."""
    return os.path.splitext(os.path.normpath(path))[1].lower()


def _load_json(path):
    """Return (content, error). The content is None if the file is absent or unreadable."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path) as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, str(e)


def classify(node_dir):
    """Classify a directory as dataset, group or defect, and return its metadata."""
    attrs, error = _load_json(os.path.join(node_dir, N5_ATTRS))
    if error is not None:
        return BROKEN, {}, f"{N5_ATTRS} is not valid JSON: {error}"
    if attrs is not None and "dimensions" in attrs:
        return DATASET, attrs, None

    if os.path.isfile(os.path.join(node_dir, ZARR2_ARRAY)):
        return ZARR_ARRAY, _load_json(os.path.join(node_dir, ZARR2_ARRAY))[0] or {}, None
    zarr3, _ = _load_json(os.path.join(node_dir, ZARR3_NODE))
    if zarr3 is not None:
        kind = ZARR_ARRAY if zarr3.get("node_type") == "array" else ZARR_GROUP
        return kind, zarr3, None
    if os.path.isfile(os.path.join(node_dir, ZARR2_GROUP)):
        return ZARR_GROUP, {}, None

    if attrs is not None:
        return GROUP, attrs, None

    entries = list(os.scandir(node_dir))
    if any(e.is_dir() and not e.name.isdigit() for e in entries):
        return IMPLICIT_GROUP, {}, None
    if any(e.name.isdigit() for e in entries):
        return NO_METADATA, {}, None
    return EMPTY_DIR, {}, None


def _child_dirs(path):
    return sorted(e.name for e in os.scandir(path) if e.is_dir())


def walk_nodes(root):
    """Yield (relative path, kind, metadata, defect) for every node, chunk dirs excluded."""
    stack = [("", root)]
    while stack:
        rel, path = stack.pop(0)
        kind, meta, defect = classify(path)
        yield rel or "/", kind, meta, defect
        # The children of a dataset are its chunk directories.
        if kind in (DATASET, ZARR_ARRAY):
            continue
        for name in _child_dirs(path):
            # Purely numeric names below a group would be chunk directories of a dataset
            # whose metadata is gone. Report the parent instead of walking the chunk grid.
            if name.isdigit() and kind == NO_METADATA:
                continue
            stack.append((f"{rel}/{name}" if rel else name, os.path.join(path, name)))


def chunk_report(dataset_dir, dimensions, block_size, max_report):
    """Compare the chunk files on disk against the grid declared in the metadata."""
    grid = [int(ceil(d / b)) for d, b in zip(dimensions, block_size)]
    present, truncated, out_of_range = set(), [], []

    for root, dirs, files in os.walk(dataset_dir):
        rel = os.path.relpath(root, dataset_dir)
        parts = [] if rel == "." else rel.split(os.sep)
        dirs[:] = [d for d in dirs if d.isdigit()]
        for name in files:
            if name == N5_ATTRS:
                continue
            if not name.isdigit() or not all(p.isdigit() for p in parts):
                continue
            index = tuple(int(p) for p in parts + [name])
            present.add(index)
            path = os.path.join(root, name)
            if os.path.getsize(path) < MIN_CHUNK_BYTES:
                truncated.append(os.path.relpath(path, dataset_dir))
            if len(index) != len(grid) or any(i >= g for i, g in zip(index, grid)):
                out_of_range.append(os.path.relpath(path, dataset_dir))

    missing = []
    for index in itertools.product(*(range(g) for g in grid)):
        if index not in present:
            missing.append(index)
            if len(missing) >= max_report:
                break
    return {
        "grid": grid, "expected": prod(grid), "present": len(present),
        "missing_examples": missing, "truncated": truncated, "out_of_range": out_of_range,
    }


def read_test(path, key):
    """Read one chunk at the origin and one at the center. Return an error message or None."""
    try:
        import z5py
    except ImportError as e:
        return f"z5py is not importable ({e})"
    try:
        with z5py.File(path, "r") as f:
            ds = f[key]
            chunks, shape = ds.chunks, ds.shape
            for name, start in (
                ("origin", (0,) * len(shape)),
                ("center", tuple((s // 2 // c) * c for s, c in zip(shape, chunks))),
            ):
                bb = tuple(slice(b, min(b + c, s)) for b, c, s in zip(start, chunks, shape))
                block = ds[bb]
                print(f"    read {name} block {tuple(block.shape)} ({block.dtype})")
    except Exception as e:  # noqa: BLE001 - the exception type is what we want to report
        return f"{type(e).__name__}: {e}"
    return None


def report_environment(path):
    """Print the reader versions and the constructor elf would use for this extension."""
    problems = []
    print("Reader environment:")
    print(f"  python {sys.version.split()[0]}")
    for name in ("z5py", "zarr", "elf"):
        try:
            module = importlib.import_module(name)
            print(f"  {name} {getattr(module, '__version__', 'unknown version')}")
        except ImportError as e:
            print(f"  {name} is not importable ({e})")

    extension = _container_extension(path)
    try:
        from elf.io.extensions import FILE_CONSTRUCTORS
    except ImportError:
        return problems
    constructor = FILE_CONSTRUCTORS.get(extension)
    print(f"  elf.io.open_file uses {constructor} for '{extension}'")
    if constructor is None:
        problems.append(f"elf.io.open_file cannot open '{extension}'. Install z5py.")
    elif extension == ".n5" and "z5py" not in f"{constructor}":
        problems.append(
            f"elf.io.open_file routes '{extension}' to {constructor}. zarr 3 cannot read n5. "
            "Install z5py and update python-elf."
        )
    return problems


def check_container(path, key=None, max_report=20, with_read=True, with_chunks=True):
    """Print the report for one container. Return the list of problems."""
    print(f"Container: {path}")
    if not os.path.isdir(path):
        print("  the path does not exist or is not a directory")
        return ["the container does not exist"]

    problems = report_environment(path)

    print("\nStructure:")
    nodes = {}
    for rel, kind, meta, defect in walk_nodes(path):
        nodes[rel] = (kind, meta)
        detail = ""
        if kind == DATASET:
            # n5 stores 'dimensions' in (x, y, z) order, z5py reports the reverse.
            dimensions, block_size = meta["dimensions"], meta.get("blockSize")
            compression = meta.get("compression", {}).get("type", meta.get("compressionType"))
            detail = (f"  dimensions (x, y, z) = {tuple(dimensions)}, blockSize = "
                      f"{tuple(block_size) if block_size else None}, "
                      f"dataType = {meta.get('dataType')}, compression = {compression}")
        elif kind == ZARR_ARRAY:
            detail = f"  shape = {tuple(meta.get('shape', ()))}, chunks/dtype from zarr metadata"
        print(f"  {rel:<40} {kind}{detail}")
        if defect is not None:
            problems.append(f"{rel}: {defect}")
        if kind == NO_METADATA:
            problems.append(
                f"{rel} holds chunk data but no '{N5_ATTRS}'. The dataset metadata was not "
                "copied, so every reader returns a group for this key. This is the signature "
                "of an incomplete copy."
            )
        elif kind == EMPTY_DIR and rel != "/":
            problems.append(f"{rel} is empty. Nothing was copied into it.")

    # The reader is picked by the file extension, so a container in the other format is
    # unreadable even though its metadata is intact.
    extension = _container_extension(path)
    kinds = {kind for kind, _ in nodes.values()}
    is_zarr, is_n5 = kinds & {ZARR_ARRAY, ZARR_GROUP}, kinds & {DATASET, GROUP}
    if extension == ".n5" and is_zarr and not is_n5:
        problems.append(
            "the container holds zarr metadata but its name ends in '.n5'. z5py reads every "
            "node of it as a group. Rename the container to '.zarr' or convert it to n5."
        )
    elif extension in (".zarr", ".zr") and is_n5 and not is_zarr:
        problems.append(
            f"the container holds n5 metadata but its name ends in '{extension}'. "
            "Rename the container to '.n5' or convert it to zarr."
        )

    key_rel = None if key is None else _normalize_key(key)
    if key_rel is not None:
        if key_rel not in nodes:
            problems.append(f"the key '{key}' does not exist in the container")
        elif nodes[key_rel][0] not in (DATASET, ZARR_ARRAY):
            problems.append(
                f"the key '{key}' does not point to an array, it is a {nodes[key_rel][0]}. "
                "Readers return a group for it, which fails as \"'Group' object has no "
                "attribute 'shape'\"."
            )

    datasets = [rel for rel, (kind, _) in nodes.items() if kind == DATASET]
    if key_rel in datasets:
        datasets = [key_rel]

    if with_chunks and datasets:
        print("\nChunks:")
        for rel in datasets:
            meta = nodes[rel][1]
            if not meta.get("blockSize"):
                problems.append(f"{rel}: '{N5_ATTRS}' has no blockSize")
                continue
            report = chunk_report(_node_dir(path, rel), meta["dimensions"],
                                  meta["blockSize"], max_report)
            missing = report["expected"] - report["present"]
            print(f"  {rel}: {report['present']}/{report['expected']} chunk files "
                  f"(grid {tuple(report['grid'])}), {len(report['truncated'])} truncated")
            if missing > 0:
                # n5 datasets may be sparse, so absent chunks are not proof of a bad copy.
                # A partial copy shows up as a large, contiguous set of missing chunks.
                print(f"    {missing} chunk file(s) absent, first: "
                      f"{report['missing_examples'][:max_report]}")
            for bad in report["truncated"][:max_report]:
                print(f"    truncated: {bad}")
                problems.append(f"{rel}: truncated chunk file {bad}")
            for bad in report["out_of_range"][:max_report]:
                print(f"    outside the declared grid: {bad}")
                problems.append(f"{rel}: chunk file outside the declared grid {bad}")

    if with_read and datasets:
        print("\nRead test:")
        for rel in datasets:
            print(f"  {rel}")
            error = read_test(path, rel)
            if error is not None:
                print(f"    failed: {error}")
                problems.append(f"{rel}: cannot be read ({error})")

    print("\nVerdict:")
    if problems:
        for problem in problems:
            print(f"  [problem] {problem}")
    else:
        print("  no problem found")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="Path to the n5 or zarr container.")
    parser.add_argument("-k", "--key", default=None,
                        help="The key that fails to open, e.g. 'setup0/timepoint0/s0'. "
                        "By default every dataset in the container is checked.")
    parser.add_argument("--max-report", type=int, default=20,
                        help="Maximum number of defective or absent chunks to list per dataset.")
    parser.add_argument("--no-read", action="store_true",
                        help="Skip reading a chunk with z5py.")
    parser.add_argument("--no-chunks", action="store_true",
                        help="Skip the chunk-file scan. Use it for a very large container.")
    args = parser.parse_args()

    problems = check_container(
        args.path, key=args.key, max_report=args.max_report,
        with_read=not args.no_read, with_chunks=not args.no_chunks,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
