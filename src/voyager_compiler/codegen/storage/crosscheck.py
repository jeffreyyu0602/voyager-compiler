# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Checking the storage graph against the partition upstream already builds.

``memory_planning._buffer_identity`` answers a narrower version of the same
question this package answers: *which nodes name one buffer*.  It answers it
for the bufferized path only, as an untyped ``Dict[Node, Node]``, and it has
been right for long enough to be worth treating as an oracle.

So the two must induce the **same partition of nodes**.  Not the same types --
this package distinguishes ``ALIAS`` from ``VIEW`` from ``WRITE_INTO`` and
``_buffer_identity`` cannot -- but if the two disagree about whether two nodes
share a backing, one of them is wrong, and on the bufferized path the safe
assumption is that it is the new one.

This runs only when ``VOYAGER_STORAGE_CROSSCHECK`` names a file to write, and
it never raises into the compile: a diagnostic that can fail a build people
depend on stops being run.
"""

import itertools
import json
import logging
import os
import traceback
from pathlib import Path

from voyager_compiler.codegen.storage.analysis import StorageGraph
from voyager_compiler.codegen.storage.types import StoragePlane, ValueRef

logger = logging.getLogger(__name__)

__all__ = ["crosscheck_buffer_identity"]

#: Names the report path; unset means the check does not run. A path ending in
#: ``/``, or naming an existing directory, collects one report per call instead
#: -- which is what makes the check usable across a whole regression, where a
#: single filename would leave only the last label's answer.
CROSSCHECK_ENV = "VOYAGER_STORAGE_CROSSCHECK"

_report_counter = itertools.count()


def _report_path(setting):
    """Where this call's report goes, given the setting."""
    path = Path(setting)
    if setting.endswith(os.sep) or path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        return path / f"crosscheck-{next(_report_counter):03d}.json"
    return path


def _root(buffer_of, node):
    """``_buffer_identity``'s representative for ``node``.

    The map is one hop, not a root: a slot ``subview`` of a ``getitem`` of a
    ``while_loop`` takes three.  Absent means "names itself", so the walk stops
    there; a cycle would too, and is reported rather than hung on.
    """
    seen = set()
    while node in buffer_of and node not in seen:
        seen.add(node)
        node = buffer_of[node]
    return node


def _labels(nodes):
    """A unique label per node, its name where that is already unique.

    Blocks are compared as sets of labels, so a label that two nodes share
    would make two different blocks look like one -- the same conflation this
    comparison exists to detect. A repeated name gets an occurrence suffix, in
    the deterministic order the nodes were walked.
    """
    counts = {}
    for node in nodes:
        counts[node.name] = counts.get(node.name, 0) + 1
    seen, labels = {}, {}
    for node in nodes:
        name = node.name
        if counts[name] == 1:
            labels[node] = name
        else:
            seen[name] = seen.get(name, 0) + 1
            labels[node] = f"{name}#{seen[name]}"
    return labels


def _partition(assignment, labels):
    """The blocks of ``{node: group key}``, each as a frozenset of labels.

    The *keys* group; the labels only name. Two group keys are never compared
    across the two sides -- one is an FX node, the other a
    ``PhysicalTensorId`` -- so what is compared is the set of blocks.
    """
    groups = {}
    for node, key in assignment.items():
        groups.setdefault(key, set()).add(labels[node])
    return {key: frozenset(members) for key, members in groups.items()}


def crosscheck_buffer_identity(model, buffer_of, plane=StoragePlane.DRAM):
    """Compare the two partitions and write a report.  Returns it, or ``None``.

    Agreement is compared as *set partitions*: the group keys are not
    comparable across the two (one is an FX node, the other a
    ``PhysicalTensorId``), but the induced blocks are.
    """
    setting = os.environ.get(CROSSCHECK_ENV)
    if not setting:
        return None
    try:
        # Once, before the comparison: in directory mode this consumes a
        # sequence number, and asking again on the failure path would file the
        # error under a different name than the run that produced it.
        path = _report_path(setting)
    except OSError:
        logger.warning("storage crosscheck report path unusable: %s", setting)
        return None
    try:
        return _compare(model, buffer_of, plane, path)
    except Exception:  # a diagnostic must not fail the build
        logger.warning("storage crosscheck failed:\n%s", traceback.format_exc())
        try:
            with open(path, "w") as handle:
                json.dump({"error": traceback.format_exc()}, handle, indent=2)
        except OSError:
            pass
        return None


def _explain(graph, plane, upstream, ours, labels, blocks, limit=12):
    """The relation this package assigned to each member of a disputed block.

    Reported with what ``_buffer_identity`` said about the same node, so the
    two answers sit side by side: ``ALIAS -> x`` against ``names itself`` is a
    merge to explain, and the transform names the rule that made it.
    """
    wanted = {label for block in blocks for label in block}
    explained = {}
    for node, identifier in ours.items():
        label = labels[node]
        if label not in wanted or len(explained) >= limit:
            continue
        try:
            relation = graph.relation(plane, ValueRef(node))
        except Exception:
            continue
        source = getattr(relation.source, "name", None)
        transform = getattr(relation.transform, "kind", None)
        explained[label] = {
            "kind": getattr(relation.kind, "value", str(relation.kind)),
            "source": source,
            "transform": transform,
            "op": getattr(node, "op", None),
            "upstream_root": upstream[node].name,
            "physical": repr(identifier),
        }
    return explained


def _compare(model, buffer_of, plane, path):
    graph = StorageGraph(model, planes=(plane,))
    plan = graph.finalize()

    # Keyed by the node *object*, never by its name. A name is not unique
    # across a graph with subgraphs -- llama's nested loop and branch bodies
    # each carry their own ``arg10_11`` -- so keying by name silently merges
    # distinct nodes and reports the merge as a disagreement in the analysis.
    # Names are for the report only.
    upstream, ours = {}, {}
    for node in buffer_of:
        ref = ValueRef(node)
        try:
            identifier = plan.physical_id(plane, ref)
        except Exception:
            continue  # not a value this plane resolves; not a disagreement
        upstream[node] = _root(buffer_of, node)
        ours[node] = identifier

    labels = _labels(list(upstream))
    theirs = _partition(upstream, labels)
    mine = _partition(ours, labels)
    their_blocks = set(theirs.values())
    my_blocks = set(mine.values())

    # A disagreement is a block one side has and the other does not. Reported
    # from both directions because they mean different things: a block only
    # upstream has is a merge this package missed (two names it left apart),
    # and a block only this package has is a split upstream does not make.
    # If names repeat, a name-keyed comparison would have been wrong; say so
    # rather than leave the reader to wonder which nodes a block names.
    names = [node.name for node in upstream]
    report = {
        "nodes_compared": len(upstream),
        "distinct_names": len(set(names)),
        "blocks_upstream": len(their_blocks),
        "blocks_storage_graph": len(my_blocks),
        "agree": their_blocks == my_blocks,
        "only_upstream": sorted(
            sorted(b) for b in their_blocks - my_blocks
        )[:50],
        "only_storage_graph": sorted(
            sorted(b) for b in my_blocks - their_blocks
        )[:50],
        # Why, not just what. A block this package has and upstream does not
        # is a merge it made; the relation that made it is the thing to read,
        # and re-running a llama compile to find out costs half an hour.
        "disagreeing_relations": _explain(
            graph, plane, upstream, ours, labels, my_blocks - their_blocks
        ),
    }
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    logger.info(
        "storage crosscheck: %d nodes, %d/%d blocks, agree=%s -> %s",
        report["nodes_compared"],
        report["blocks_upstream"],
        report["blocks_storage_graph"],
        report["agree"],
        path,
    )
    return report
