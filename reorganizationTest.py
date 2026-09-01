from src.simulator.heap_allocator import HeapAllocator
from src.controller.policies import CostAwarePredictivePolicy

# 1. Initialize heap and proactive cost-aware controller
allocator = HeapAllocator(total_capacity=2 * 1024 * 1024)
policy = CostAwarePredictivePolicy(benefit_scale_lambda=1500.0, min_net_benefit_threshold=2.0)

# 2. Simulate allocations & frees creating holes
a1 = allocator.allocate(512 * 1024)
a2 = allocator.allocate(256 * 1024)
a3 = allocator.allocate(512 * 1024)
allocator.free(a2) # Creates a hole

# 3. If model predicts high future fragmentation (e.g., F1 = 0.75):
predicted_f1_future = 0.75
should_reorganize = policy.decide(
    tick=10, 
    current_f1=allocator.get_snapshot().f1_largest_free_ratio, 
    predicted_f1=predicted_f1_future, 
    blocks=allocator.blocks
)

if should_reorganize:
    report = allocator.compact(timestamp=10.0)
    print(f"[Proactive Compaction Triggered] Blocks moved: {report.blocks_moved}, Cost: {report.total_cost:.2f}, F1 after: {report.f1_after}")