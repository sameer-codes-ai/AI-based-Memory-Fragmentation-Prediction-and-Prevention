"""
PREFRAG: Workload Trace Generator
Generates realistic, reproducible allocation/deallocation traces across diverse benchmark families.
"""

from enum import Enum
import random
import math
from typing import List, Tuple, Optional
from .trace_schema import MemoryEvent, AllocationOp


class WorkloadFamily(str, Enum):
    WEB_SERVER = "web_server"
    DATABASE = "database"
    SCIENTIFIC = "scientific"
    MICROSERVICE = "microservice"
    ADVERSARIAL_CHURN = "adversarial_churn"


class WorkloadGenerator:
    """
    Generates deterministic memory allocation traces simulating realistic application workloads.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self._address_counter = 0x10000000  # Base heap virtual address
        self._ptr_map = {} # virtual id to size/lifetime

    def _next_address(self, size: int) -> int:
        addr = self._address_counter
        self._address_counter += max(16, ((size + 15) // 16) * 16)
        return addr

    def generate_trace(
        self,
        family: WorkloadFamily,
        num_events: int = 5000,
        time_step_ms: float = 1.0,
        tag_prefix: str = ""
    ) -> List[MemoryEvent]:
        """
        Generates a sequence of MemoryEvents for the specified workload family.
        """
        self.rng.seed(self.seed)
        self._address_counter = 0x10000000

        if family == WorkloadFamily.WEB_SERVER:
            return self._generate_web_server(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.DATABASE:
            return self._generate_database(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.SCIENTIFIC:
            return self._generate_scientific(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.MICROSERVICE:
            return self._generate_microservice(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.ADVERSARIAL_CHURN:
            return self._generate_adversarial(num_events, time_step_ms, tag_prefix)
        else:
            raise ValueError(f"Unknown workload family: {family}")

    def _generate_web_server(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates Nginx / Redis / Django server workloads:
        - Bursty request arrival
        - Many small request buffers (64B - 2KB) with short lifetime
        - Some medium response caches (4KB - 64KB) with longer lifetime
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = [] # (free_time, address, size, call_stack)
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "web_server"

        while event_count < num_events:
            curr_time += self.rng.expovariate(1.0 / time_step_ms)
            
            # Check and process pending frees that expired before curr_time
            pending_frees = [item for item in active_allocations if item[0] <= curr_time]
            for free_time, addr, sz, c_hash in sorted(pending_frees, key=lambda x: x[0]):
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                active_allocations.remove((free_time, addr, sz, c_hash))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            # Generate new allocation
            # 80% small requests, 20% cached responses
            if self.rng.random() < 0.80:
                size = int(self.rng.lognormvariate(math.log(256), 0.7))
                size = max(32, min(size, 4096))
                lifetime = self.rng.uniform(5.0 * time_step_ms, 25.0 * time_step_ms)
                call_stack = 0x1001
            else:
                size = int(self.rng.lognormvariate(math.log(16384), 0.8))
                size = max(4096, min(size, 65536))
                lifetime = self.rng.uniform(50.0 * time_step_ms, 300.0 * time_step_ms)
                call_stack = 0x2002

            addr = self._next_address(size)
            events.append(MemoryEvent(
                timestamp=curr_time,
                op=AllocationOp.ALLOC,
                address=addr,
                size=size,
                lifetime=lifetime,
                call_stack_hash=call_stack,
                tag=tag_str
            ))
            active_allocations.append((curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        # Drain remaining frees in timestamp order
        for free_time, addr, sz, c_hash in sorted(active_allocations, key=lambda x: x[0]):
            events.append(MemoryEvent(
                timestamp=free_time,
                op=AllocationOp.FREE,
                address=addr,
                size=sz,
                call_stack_hash=c_hash,
                tag=tag_str
            ))

        events.sort(key=lambda e: e.timestamp)
        return events[:num_events]

    def _generate_database(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates SQLite / RocksDB / Buffer Pool workloads:
        - Fixed-size page allocations (4KB, 8KB, 16KB)
        - Long-lived table buffer pools mixed with short-lived transaction query memory
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "database"

        while event_count < num_events:
            curr_time += time_step_ms
            
            pending_frees = [item for item in active_allocations if item[0] <= curr_time]
            for free_time, addr, sz, c_hash in sorted(pending_frees, key=lambda x: x[0]):
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                active_allocations.remove((free_time, addr, sz, c_hash))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            r = self.rng.random()
            if r < 0.60:
                # 4KB/8KB Page allocation (long to medium lifespan)
                size = self.rng.choice([4096, 8192, 16384])
                lifetime = self.rng.uniform(100.0 * time_step_ms, 800.0 * time_step_ms)
                call_stack = 0xDB01
            elif r < 0.85:
                # Short-lived transaction query workspace (512B - 32KB)
                size = int(self.rng.uniform(512, 32768))
                lifetime = self.rng.uniform(10.0 * time_step_ms, 40.0 * time_step_ms)
                call_stack = 0xDB02
            else:
                # Index key/value node allocation (64B - 512B)
                size = int(self.rng.uniform(64, 512))
                lifetime = self.rng.uniform(50.0 * time_step_ms, 200.0 * time_step_ms)
                call_stack = 0xDB03

            addr = self._next_address(size)
            events.append(MemoryEvent(
                timestamp=curr_time,
                op=AllocationOp.ALLOC,
                address=addr,
                size=size,
                lifetime=lifetime,
                call_stack_hash=call_stack,
                tag=tag_str
            ))
            active_allocations.append((curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        for free_time, addr, sz, c_hash in sorted(active_allocations, key=lambda x: x[0]):
            events.append(MemoryEvent(
                timestamp=free_time,
                op=AllocationOp.FREE,
                address=addr,
                size=sz,
                call_stack_hash=c_hash,
                tag=tag_str
            ))

        events.sort(key=lambda e: e.timestamp)
        return events[:num_events]

    def _generate_scientific(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates Scientific / Matrix / PyTorch iteration loops:
        - Phased allocation cycles (epoch / batch compute cycles)
        - Large matrices / tensor buffers (64KB - 512KB)
        - Coordinated synchronous frees at phase boundaries
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "scientific"
        phase_length = 50 # events per compute phase

        while event_count < num_events:
            phase_id = (event_count // phase_length) % 5
            curr_time += time_step_ms

            pending_frees = [item for item in active_allocations if item[0] <= curr_time]
            for free_time, addr, sz, c_hash in sorted(pending_frees, key=lambda x: x[0]):
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                active_allocations.remove((free_time, addr, sz, c_hash))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            # Phase-dependent size and lifecycle
            if phase_id in (0, 1): # Forward pass: large activations
                size = int(self.rng.uniform(32768, 262144))
                lifetime = float(phase_length * time_step_ms)
                call_stack = 0x5C01
            elif phase_id == 2: # Loss / reduce: small accumulators
                size = int(self.rng.uniform(128, 2048))
                lifetime = float(5.0 * time_step_ms)
                call_stack = 0x5C02
            else: # Backward pass: gradient and optimizer states
                size = int(self.rng.uniform(65536, 524288))
                lifetime = float(phase_length * 2 * time_step_ms)
                call_stack = 0x5C03

            addr = self._next_address(size)
            events.append(MemoryEvent(
                timestamp=curr_time,
                op=AllocationOp.ALLOC,
                address=addr,
                size=size,
                lifetime=lifetime,
                call_stack_hash=call_stack,
                tag=tag_str
            ))
            active_allocations.append((curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        for free_time, addr, sz, c_hash in sorted(active_allocations, key=lambda x: x[0]):
            events.append(MemoryEvent(
                timestamp=free_time,
                op=AllocationOp.FREE,
                address=addr,
                size=sz,
                call_stack_hash=c_hash,
                tag=tag_str
            ))

        events.sort(key=lambda e: e.timestamp)
        return events[:num_events]

    def _generate_microservice(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates Microservice / JSON / RPC parser stream:
        - High frequency allocation of variable strings/objects
        - Random out-of-order deallocation inducing severe spatial fragmentation
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "microservice"

        while event_count < num_events:
            curr_time += time_step_ms * self.rng.uniform(0.5, 1.5)

            pending_frees = [item for item in active_allocations if item[0] <= curr_time]
            for free_time, addr, sz, c_hash in sorted(pending_frees, key=lambda x: x[0]):
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                active_allocations.remove((free_time, addr, sz, c_hash))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            # Power-law distributed string / JSON token sizes
            size = int(self.rng.paretovariate(1.5) * 64)
            size = max(16, min(size, 32768))
            lifetime = self.rng.exponential(20.0 * time_step_ms) if hasattr(self.rng, 'exponential') else self.rng.expovariate(1.0 / (20.0 * time_step_ms))
            lifetime = max(2.0 * time_step_ms, lifetime)
            call_stack = 0xE501

            addr = self._next_address(size)
            events.append(MemoryEvent(
                timestamp=curr_time,
                op=AllocationOp.ALLOC,
                address=addr,
                size=size,
                lifetime=lifetime,
                call_stack_hash=call_stack,
                tag=tag_str
            ))
            active_allocations.append((curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        for free_time, addr, sz, c_hash in sorted(active_allocations, key=lambda x: x[0]):
            events.append(MemoryEvent(
                timestamp=free_time,
                op=AllocationOp.FREE,
                address=addr,
                size=sz,
                call_stack_hash=c_hash,
                tag=tag_str
            ))

        events.sort(key=lambda e: e.timestamp)
        return events[:num_events]

    def _generate_adversarial(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Alternating pinned and transient allocations specifically crafted to induce checkerboarding.
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "adversarial_churn"

        step = 0
        while event_count < num_events:
            curr_time += time_step_ms
            step += 1

            pending_frees = [item for item in active_allocations if item[0] <= curr_time]
            for free_time, addr, sz, c_hash in sorted(pending_frees, key=lambda x: x[0]):
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                active_allocations.remove((free_time, addr, sz, c_hash))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            if step % 2 == 0:
                # Transient block: freed quickly
                size = 1024
                lifetime = 5.0 * time_step_ms
                call_stack = 0xAD01
            else:
                # Pinned block: persistent
                size = 512
                lifetime = 500.0 * time_step_ms
                call_stack = 0xAD02

            addr = self._next_address(size)
            events.append(MemoryEvent(
                timestamp=curr_time,
                op=AllocationOp.ALLOC,
                address=addr,
                size=size,
                lifetime=lifetime,
                call_stack_hash=call_stack,
                tag=tag_str
            ))
            active_allocations.append((curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        for free_time, addr, sz, c_hash in sorted(active_allocations, key=lambda x: x[0]):
            events.append(MemoryEvent(
                timestamp=free_time,
                op=AllocationOp.FREE,
                address=addr,
                size=sz,
                call_stack_hash=c_hash,
                tag=tag_str
            ))

        events.sort(key=lambda e: e.timestamp)
        return events[:num_events]
