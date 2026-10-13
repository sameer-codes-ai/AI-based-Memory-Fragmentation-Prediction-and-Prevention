"""
PREFRAG: Workload Trace Generator
Generates realistic, reproducible allocation/deallocation traces across diverse benchmark families.
"""

from enum import Enum
import random
import math
import heapq
from typing import List, Tuple, Optional
from .trace_schema import MemoryEvent, AllocationOp


class WorkloadFamily(str, Enum):
    WEB_SERVER = "web_server"
    DATABASE = "database"
    SCIENTIFIC = "scientific"
    MICROSERVICE = "microservice"
    ADVERSARIAL_CHURN = "adversarial_churn"
    EMBEDDED_IOT = "embedded_iot"
    GAME_ENGINE = "game_engine"
    COMPILER = "compiler"
    STREAMING_MEDIA = "streaming_media"
    GARBAGE_COLLECTED = "garbage_collected"


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
        elif family == WorkloadFamily.EMBEDDED_IOT:
            return self._generate_embedded_iot(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.GAME_ENGINE:
            return self._generate_game_engine(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.COMPILER:
            return self._generate_compiler(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.STREAMING_MEDIA:
            return self._generate_streaming_media(num_events, time_step_ms, tag_prefix)
        elif family == WorkloadFamily.GARBAGE_COLLECTED:
            return self._generate_garbage_collected(num_events, time_step_ms, tag_prefix)
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
            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        # Drain remaining frees in timestamp order
        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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
            
            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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
        Alternating pinned and transient allocations crafted to induce spatial checkerboarding.
        Includes stochastic size perturbation (+/-20%) and multiple call stacks.
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

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            if step % 2 == 0:
                # Transient block: freed quickly with size jitter
                base_size = 1024
                jitter = int(self.rng.uniform(-200, 300))
                size = max(128, base_size + jitter)
                lifetime = self.rng.uniform(3.0 * time_step_ms, 8.0 * time_step_ms)
                call_stack = 0xAD01 + (step % 6)
            else:
                # Pinned block: persistent with size jitter
                base_size = 512
                jitter = int(self.rng.uniform(-100, 150))
                size = max(64, base_size + jitter)
                lifetime = self.rng.uniform(300.0 * time_step_ms, 800.0 * time_step_ms)
                call_stack = 0xAD10 + (step % 6)

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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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

    def _generate_embedded_iot(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates RTOS / Embedded IoT firmware:
        - Small, constrained allocations (32B - 512B)
        - High proportion of sensor reading buffers with short lifetimes
        - Long-lived device state and peripheral configurations
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "embedded_iot"

        while event_count < num_events:
            curr_time += time_step_ms * self.rng.uniform(0.8, 1.2)

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            r = self.rng.random()
            if r < 0.70:
                # Sensor telemetry buffer (short-lived, 32B - 128B)
                size = self.rng.choice([32, 48, 64, 96, 128])
                lifetime = self.rng.uniform(5.0 * time_step_ms, 20.0 * time_step_ms)
                call_stack = 0xEE01 + (event_count % 4)
            elif r < 0.90:
                # Network packet frame buffer (medium-lived, 256B - 512B)
                size = self.rng.choice([256, 384, 512])
                lifetime = self.rng.uniform(30.0 * time_step_ms, 120.0 * time_step_ms)
                call_stack = 0xEE10 + (event_count % 4)
            else:
                # Persistent driver/config block (long-lived)
                size = int(self.rng.uniform(128, 512))
                lifetime = self.rng.uniform(500.0 * time_step_ms, 2000.0 * time_step_ms)
                call_stack = 0xEE20 + (event_count % 3)

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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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

    def _generate_game_engine(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates Game Engine frame loop (60 FPS):
        - Bimodal allocation: high-volume ephemeral frame entities + large persistent assets
        - Frame boundaries produce bursts of deallocations
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "game_engine"
        frame_time = 16.6 * time_step_ms

        while event_count < num_events:
            curr_time += time_step_ms

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            r = self.rng.random()
            if r < 0.85:
                # Per-frame entity/particle/matrix allocation (16B - 1024B, dies within frame)
                size = int(self.rng.uniform(16, 1024))
                lifetime = self.rng.uniform(2.0 * time_step_ms, frame_time)
                call_stack = 0xCA01 + (event_count % 8)
            elif r < 0.97:
                # Sound clip or animation track (4KB - 32KB, medium lifetime)
                size = int(self.rng.uniform(4096, 32768))
                lifetime = self.rng.uniform(frame_time * 5, frame_time * 30)
                call_stack = 0xCA10 + (event_count % 4)
            else:
                # High-res mesh / texture staging buffer (64KB - 512KB, persistent)
                size = int(self.rng.uniform(65536, 524288))
                lifetime = self.rng.uniform(frame_time * 50, frame_time * 200)
                call_stack = 0xCA20 + (event_count % 3)

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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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

    def _generate_compiler(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates Compiler pass stages (Lexer -> Parser/AST -> IR Optimization -> CodeGen):
        - Phase-dependent allocation size and lifetime distributions
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "compiler"
        phase_size = 100

        while event_count < num_events:
            curr_time += time_step_ms
            phase = (event_count // phase_size) % 4

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            if phase == 0:
                # Lexer token stream: small, short-lived (32B - 128B)
                size = int(self.rng.uniform(32, 128))
                lifetime = self.rng.uniform(5.0 * time_step_ms, 25.0 * time_step_ms)
                call_stack = 0xCC01 + (event_count % 4)
            elif phase == 1:
                # AST syntax nodes: medium, lives until semantic pass (64B - 1024B)
                size = int(self.rng.uniform(64, 1024))
                lifetime = self.rng.uniform(50.0 * time_step_ms, 150.0 * time_step_ms)
                call_stack = 0xCC10 + (event_count % 5)
            elif phase == 2:
                # IR optimization SSA structures (512B - 8192B)
                size = int(self.rng.uniform(512, 8192))
                lifetime = self.rng.uniform(80.0 * time_step_ms, 250.0 * time_step_ms)
                call_stack = 0xCC20 + (event_count % 4)
            else:
                # CodeGen / binary emission buffer (4KB - 64KB, lives until file flush)
                size = int(self.rng.uniform(4096, 65536))
                lifetime = self.rng.uniform(150.0 * time_step_ms, 500.0 * time_step_ms)
                call_stack = 0xCC30 + (event_count % 3)

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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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

    def _generate_streaming_media(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates Video / Audio streaming buffer pool:
        - Uniform chunk allocations (16KB, 32KB, 64KB)
        - Sliding deallocation window mimicking FIFO playback consumption
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "streaming_media"

        while event_count < num_events:
            curr_time += time_step_ms

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            # 70% standard media chunk, 20% audio slice, 10% metadata header
            r = self.rng.random()
            if r < 0.70:
                size = self.rng.choice([32768, 65536, 131072])
                lifetime = self.rng.uniform(50.0 * time_step_ms, 200.0 * time_step_ms)
                call_stack = 0xBA01 + (event_count % 3)
            elif r < 0.90:
                size = self.rng.choice([4096, 8192, 16384])
                lifetime = self.rng.uniform(20.0 * time_step_ms, 80.0 * time_step_ms)
                call_stack = 0xBA10 + (event_count % 3)
            else:
                size = int(self.rng.uniform(256, 2048))
                lifetime = self.rng.uniform(200.0 * time_step_ms, 800.0 * time_step_ms)
                call_stack = 0xBA20 + (event_count % 2)

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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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

    def _generate_garbage_collected(self, num_events: int, time_step_ms: float, tag: str) -> List[MemoryEvent]:
        """
        Simulates Managed Heap (JVM / Go GC) generational lifecycle:
        - Strong generational hypothesis: 85% die young in nursery
        - 15% promoted to tenured generation with long persistence
        """
        events: List[MemoryEvent] = []
        active_allocations: List[Tuple[float, int, int, int]] = []
        curr_time = 0.0
        event_count = 0
        tag_str = tag or "garbage_collected"

        while event_count < num_events:
            curr_time += time_step_ms * self.rng.uniform(0.7, 1.3)

            while active_allocations and active_allocations[0][0] <= curr_time:
                free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
                events.append(MemoryEvent(
                    timestamp=free_time,
                    op=AllocationOp.FREE,
                    address=addr,
                    size=sz,
                    call_stack_hash=c_hash,
                    tag=tag_str
                ))
                event_count += 1
                if event_count >= num_events:
                    break

            if event_count >= num_events:
                break

            if self.rng.random() < 0.85:
                # Nursery object: power-law size, short lifetime (dies before next GC tick)
                size = int(self.rng.paretovariate(1.8) * 48)
                size = max(16, min(size, 8192))
                lifetime = self.rng.uniform(3.0 * time_step_ms, 15.0 * time_step_ms)
                call_stack = 0xDA01 + (event_count % 6)
            else:
                # Promoted / Tenured object: survives multiple cycles
                size = int(self.rng.uniform(2048, 65536))
                lifetime = self.rng.uniform(300.0 * time_step_ms, 1500.0 * time_step_ms)
                call_stack = 0xDA10 + (event_count % 4)

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
            heapq.heappush(active_allocations, (curr_time + lifetime, addr, size, call_stack))
            event_count += 1

        while active_allocations:
            free_time, addr, sz, c_hash = heapq.heappop(active_allocations)
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
