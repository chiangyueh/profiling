"""Hardware parameters used by the generic NPU cycle simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import MemorySpace, Resource


@dataclass(frozen=True)
class ResourceRate:
    bytes_per_cycle: float = 0.0
    operations_per_cycle: float = 0.0
    issue_cycles: float = 0.0
    latency_cycles: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.bytes_per_cycle,
            self.operations_per_cycle,
            self.issue_cycles,
            self.latency_cycles,
        ) < 0.0:
            raise ValueError("hardware rates must be non-negative")


@dataclass(frozen=True)
class Hardware:
    name: str
    core_counts: dict[Resource, int]
    rates: dict[Resource, ResourceRate]
    capacities: dict[MemorySpace, int]
    parallel_units: dict[Resource, float] = field(default_factory=dict)
    dtype_operation_rates: dict[tuple[Resource, str], float] = field(
        default_factory=dict
    )
    route_byte_rates: dict[tuple[MemorySpace, MemorySpace], float] = field(
        default_factory=dict
    )
    aggregate_hbm_bytes_per_cycle: float = 0.0
    aggregate_l2_bytes_per_cycle: float = 0.0
    kernel_launch_cycles: float = 0.0
    active_core_launch_cycles: float = 0.0
    transaction_bytes: int = 32

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.core_counts.values()):
            raise ValueError("hardware core counts must be positive")
        if any(value < 0 for value in self.capacities.values()):
            raise ValueError("memory capacities must be non-negative")
        if self.transaction_bytes <= 0:
            raise ValueError("transaction_bytes must be positive")

    def core_count(self, resource: Resource) -> int:
        return self.core_counts.get(resource, 1)

    def rate(self, resource: Resource, dtype: str | None = None) -> ResourceRate:
        base = self.rates.get(resource, ResourceRate())
        if dtype is None:
            return base
        operations = self.dtype_operation_rates.get(
            (resource, dtype), base.operations_per_cycle
        )
        return ResourceRate(
            bytes_per_cycle=base.bytes_per_cycle,
            operations_per_cycle=operations,
            issue_cycles=base.issue_cycles,
            latency_cycles=base.latency_cycles,
        )

    def route_rate(self, source: MemorySpace, destination: MemorySpace) -> float:
        return self.route_byte_rates.get((source, destination), 0.0)


def ascend_910b3() -> Hardware:
    """Frozen Ascend 910B3 primitive profile.

    Values are architectural capacities or isolated CCE primitive service
    measurements already used by the repository's C++ hardware evaluator.
    Runtime scoring reads no calibration file or operator latency history.
    Cube rates count multiply-accumulate operations, not FLOPs.
    """

    return Hardware(
        name="Ascend910B3",
        core_counts={
            Resource.CUBE: 20,
            Resource.VECTOR: 40,
            Resource.SCALAR: 40,
            Resource.ATOMIC: 40,
        },
        rates={
            Resource.MTE2: ResourceRate(16.0, 0.0, 10.0, 337.0),
            Resource.MTE1: ResourceRate(256.0, 0.0, 2.0, 0.0),
            Resource.MTE3: ResourceRate(64.0, 0.0, 10.0, 25.0),
            # A C220 ``mad`` command has a 21-cycle architectural startup
            # followed by M*N*K/4096 FP16/BF16 service.  The value is the
            # frozen isolated-intrinsic intercept already established by the
            # repository's CCE primitive campaign; no table is read while a
            # tiling is scored.
            Resource.CUBE: ResourceRate(0.0, 0.0, 1.0, 21.0),
            Resource.VECTOR: ResourceRate(0.0, 128.0, 1.0, 0.0),
            Resource.SCALAR: ResourceRate(0.0, 1.0, 1.0, 0.0),
            Resource.FIXPIPE: ResourceRate(64.0, 0.0, 31.0, 0.0),
            # Until an isolated atomic CCE probe is available, use the
            # conservative serialized read-modify-write composition of MTE2
            # and MTE3: 1/(1/16+1/64)=12.8 B/cycle, with both issue and
            # completion costs.  This is a hardware-path derivation, not an
            # operator-latency fit.
            Resource.ATOMIC: ResourceRate(12.8, 0.0, 20.0, 362.0),
            Resource.SYNC: ResourceRate(0.0, 64.0 / 545.0, 1.0, 0.0),
        },
        capacities={
            MemorySpace.L0A: 64 * 1024,
            MemorySpace.L0B: 64 * 1024,
            MemorySpace.L0C: 128 * 1024,
            MemorySpace.L1: 524032,
            MemorySpace.UB: 196352,
            MemorySpace.L2: 192 * 1024 * 1024,
        },
        parallel_units={
            Resource.MTE2: 16.0,
            Resource.MTE3: 16.0,
            Resource.VECTOR: 40.0,
            Resource.SCALAR: 40.0,
            Resource.ATOMIC: 16.0,
        },
        dtype_operation_rates={
            (Resource.CUBE, "fp16"): 4096.0,
            (Resource.CUBE, "bf16"): 4096.0,
            (Resource.CUBE, "fp32"): 2048.0,
            (Resource.VECTOR, "fp16"): 128.0,
            (Resource.VECTOR, "bf16"): 128.0,
            (Resource.VECTOR, "fp32"): 64.0,
            (Resource.VECTOR, "int32"): 64.0,
            (Resource.SCALAR, "int32"): 1.0,
            (Resource.SCALAR, "int64"): 1.0,
        },
        route_byte_rates={
            (MemorySpace.GM, MemorySpace.L1): 16.0,
            (MemorySpace.GM, MemorySpace.UB): 16.0,
            # A resident L2 line still crosses the per-core ingress port.
            # The isolated C220 profile sustains about 110 B/cycle per AIC;
            # this is distinct from both a 16 B/cycle HBM miss and a free
            # local-memory reuse.
            (MemorySpace.L2, MemorySpace.L1): 110.0,
            (MemorySpace.L2, MemorySpace.UB): 110.0,
            (MemorySpace.L1, MemorySpace.L0A): 256.0,
            (MemorySpace.L1, MemorySpace.L0B): 256.0,
            (MemorySpace.L0C, MemorySpace.GM): 64.0,
            (MemorySpace.UB, MemorySpace.GM): 64.0,
        },
        aggregate_hbm_bytes_per_cycle=20.0 * 32.0,
        aggregate_l2_bytes_per_cycle=20.0 * 110.0,
        kernel_launch_cycles=96.0,
        active_core_launch_cycles=0.0,
        transaction_bytes=32,
    )
