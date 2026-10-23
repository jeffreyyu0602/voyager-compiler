# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Logical and physical storage identity.

:mod:`.types` is the vocabulary; :mod:`.effects` derives it from what the
operations mean. Nothing here decides an address -- that is the allocator's,
and keeping the two apart is what lets a plan be checked before one exists.
"""

from voyager_compiler.codegen.storage.analysis import (
    GRAPH_SCOPE_OUTER_FX,
    GRAPH_SCOPE_RECURSIVE,
    GRAPH_SCOPES,
    GraphIndex,
    StorageGraph,
    StoragePlan,
)
from voyager_compiler.codegen.storage.effects import generic_storage_relation
from voyager_compiler.codegen.storage.types import (
    LogicalTensor,
    PhysicalAccess,
    PhysicalPayloadRef,
    PhysicalTensor,
    PhysicalTensorComponent,
    PhysicalTensorId,
    SerializedTensorArtifact,
    SerializedTensorRef,
    StorageInterval,
    StoragePlane,
    StorageRelation,
    StorageRelationKind,
    StorageTransfer,
    ValueRef,
    ViewTransform,
)

__all__ = [
    "LogicalTensor",
    "GRAPH_SCOPE_OUTER_FX",
    "GRAPH_SCOPE_RECURSIVE",
    "GRAPH_SCOPES",
    "GraphIndex",
    "StorageGraph",
    "StoragePlan",
    "generic_storage_relation",
    "PhysicalAccess",
    "PhysicalPayloadRef",
    "PhysicalTensor",
    "PhysicalTensorComponent",
    "PhysicalTensorId",
    "SerializedTensorArtifact",
    "SerializedTensorRef",
    "StorageInterval",
    "StoragePlane",
    "StorageRelation",
    "StorageRelationKind",
    "StorageTransfer",
    "ValueRef",
    "ViewTransform",
]
