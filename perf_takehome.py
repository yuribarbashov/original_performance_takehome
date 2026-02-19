"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        def rw(engine, slot):
            reads = set()
            writes = set()
            mem_read = False
            mem_write = False

            if engine == "alu":
                _, dest, a1, a2 = slot
                writes.add(dest)
                reads.add(a1)
                reads.add(a2)
            elif engine == "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        writes.update(range(dest, dest + VLEN))
                        reads.add(src)
                    case ("multiply_add", dest, a, b, c):
                        writes.update(range(dest, dest + VLEN))
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                        reads.update(range(c, c + VLEN))
                    case (_, dest, a1, a2):
                        writes.update(range(dest, dest + VLEN))
                        reads.update(range(a1, a1 + VLEN))
                        reads.update(range(a2, a2 + VLEN))
            elif engine == "load":
                match slot:
                    case ("load", dest, addr):
                        writes.add(dest)
                        reads.add(addr)
                        mem_read = True
                    case ("load_offset", dest, addr, offset):
                        writes.add(dest + offset)
                        reads.add(addr + offset)
                        mem_read = True
                    case ("vload", dest, addr):
                        writes.update(range(dest, dest + VLEN))
                        reads.add(addr)
                        mem_read = True
                    case ("const", dest, _):
                        writes.add(dest)
            elif engine == "store":
                match slot:
                    case ("store", addr, src):
                        reads.add(addr)
                        reads.add(src)
                        mem_write = True
                    case ("vstore", addr, src):
                        reads.add(addr)
                        reads.update(range(src, src + VLEN))
                        mem_write = True
            elif engine == "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        writes.add(dest)
                        reads.update([cond, a, b])
                    case ("add_imm", dest, a, _):
                        writes.add(dest)
                        reads.add(a)
                    case ("vselect", dest, cond, a, b):
                        writes.update(range(dest, dest + VLEN))
                        reads.update(range(cond, cond + VLEN))
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                    case ("trace_write", val):
                        reads.add(val)
                    case ("cond_jump", cond, _):
                        reads.add(cond)
                    case ("cond_jump_rel", cond, _):
                        reads.add(cond)
                    case ("jump_indirect", addr):
                        reads.add(addr)
                    case ("coreid", dest):
                        writes.add(dest)
                    case _:
                        pass

            return reads, writes, mem_read, mem_write

        instrs = []
        cur = {}
        cur_reads = set()
        cur_writes = set()
        cur_mem_read = False
        cur_mem_write = False

        def flush():
            nonlocal cur, cur_reads, cur_writes, cur_mem_read, cur_mem_write
            if cur:
                instrs.append(cur)
            cur = {}
            cur_reads = set()
            cur_writes = set()
            cur_mem_read = False
            cur_mem_write = False

        for engine, slot in slots:
            reads, writes, mem_read, mem_write = rw(engine, slot)
            engine_slots = cur.get(engine, [])
            limit_reached = len(engine_slots) >= SLOT_LIMITS[engine]
            dep_conflict = bool(writes & (cur_reads | cur_writes)) or bool(
                reads & cur_writes
            )
            mem_conflict = (mem_write and cur_mem_read) or (mem_read and cur_mem_write)

            if limit_reached or dep_conflict or mem_conflict:
                flush()

            cur.setdefault(engine, []).append(slot)
            cur_reads |= reads
            cur_writes |= writes
            cur_mem_read = cur_mem_read or mem_read
            cur_mem_write = cur_mem_write or mem_write

        flush()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel using scratch-resident state + SIMD VALU on blocks.
        """
        tmp_init = self.alloc_scratch("tmp_init")
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_init, i))
            self.add("load", ("load", self.scratch[v], tmp_init))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        idx_base = self.alloc_scratch("idx_cache", batch_size)
        val_base = self.alloc_scratch("val_cache", batch_size)

        # Vector temporaries/constants
        max_inflight_blocks = 6
        wave_count = 1
        node_pool = self.alloc_scratch("node_pool", wave_count * max_inflight_blocks * VLEN)
        t1_pool = self.alloc_scratch("t1_pool", wave_count * max_inflight_blocks * VLEN)
        t2_pool = self.alloc_scratch("t2_pool", wave_count * max_inflight_blocks * VLEN)
        t3_pool = self.alloc_scratch("t3_pool", wave_count * max_inflight_blocks * VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)

        # Scalar lanes for gather addresses
        addr_pool = [
            self.alloc_scratch(f"addr_lane_{w}_{bi}_{lane}")
            for w in range(wave_count)
            for bi in range(max_inflight_blocks)
            for lane in range(VLEN)
        ]

        # Scalar temporaries for tail path
        tail_node = self.alloc_scratch("tail_node")
        tail_t1 = self.alloc_scratch("tail_t1")
        tail_t2 = self.alloc_scratch("tail_t2")
        tail_t3 = self.alloc_scratch("tail_t3")

        # Broadcast scalar constants used by vector path.
        self.add("valu", ("vbroadcast", zero_vec, zero_const))
        self.add("valu", ("vbroadcast", one_vec, one_const))
        self.add("valu", ("vbroadcast", two_vec, two_const))
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))

        # Stage-specific constant vectors (pre-broadcast once).
        stage_meta = []
        for si, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = self.alloc_scratch(f"hash_mul_{si}", VLEN)
                add = self.alloc_scratch(f"hash_add_{si}", VLEN)
                self.add("valu", ("vbroadcast", mul, self.scratch_const((1 << val3) + 1)))
                self.add("valu", ("vbroadcast", add, self.scratch_const(val1)))
                stage_meta.append(("mad", mul, add))
            else:
                c1 = self.alloc_scratch(f"hash_c1_{si}", VLEN)
                c3 = self.alloc_scratch(f"hash_c3_{si}", VLEN)
                self.add("valu", ("vbroadcast", c1, self.scratch_const(val1)))
                self.add("valu", ("vbroadcast", c3, self.scratch_const(val3)))
                stage_meta.append(("std", op1, op2, op3, c1, c3))

        body = []

        # Preload indices/values into scratch caches.
        full_blocks = batch_size // VLEN
        tail_start = full_blocks * VLEN
        for b in range(full_blocks):
            i = b * VLEN
            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tail_t1, self.scratch["inp_indices_p"], i_const)))
            body.append(("load", ("vload", idx_base + i, tail_t1)))
            body.append(("alu", ("+", tail_t1, self.scratch["inp_values_p"], i_const)))
            body.append(("load", ("vload", val_base + i, tail_t1)))
        for i in range(tail_start, batch_size):
            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tail_t1, self.scratch["inp_indices_p"], i_const)))
            body.append(("load", ("load", idx_base + i, tail_t1)))
            body.append(("alu", ("+", tail_t1, self.scratch["inp_values_p"], i_const)))
            body.append(("load", ("load", val_base + i, tail_t1)))

        def emit_window_gather(win, wave):
            ops = []
            wb = win * max_inflight_blocks
            active = min(max_inflight_blocks, full_blocks - wb)
            if active <= 0:
                return ops
            wave_off = wave * max_inflight_blocks * VLEN
            addr_off = wave * max_inflight_blocks * VLEN
            for lane in range(VLEN):
                for bi in range(active):
                    block = wb + bi
                    i = block * VLEN
                    addr = addr_pool[addr_off + bi * VLEN + lane]
                    ops.append(("alu", ("+", addr, self.scratch["forest_values_p"], idx_base + i + lane)))
            for lane in range(VLEN):
                for bi in range(active):
                    addr = addr_pool[addr_off + bi * VLEN + lane]
                    ops.append(("load", ("load", node_pool + wave_off + bi * VLEN + lane, addr)))
            return ops

        def emit_window_compute(win, wave):
            ops = []
            wb = win * max_inflight_blocks
            active = min(max_inflight_blocks, full_blocks - wb)
            if active <= 0:
                return ops
            wave_off = wave * max_inflight_blocks * VLEN
            for bi in range(active):
                i = (wb + bi) * VLEN
                ops.append(("valu", ("^", val_base + i, val_base + i, node_pool + wave_off + bi * VLEN)))
            for stage in stage_meta:
                if stage[0] == "mad":
                    _, mul, add = stage
                    for bi in range(active):
                        i = (wb + bi) * VLEN
                        ops.append(("valu", ("multiply_add", val_base + i, val_base + i, mul, add)))
                else:
                    _, op1, op2, op3, c1, c3 = stage
                    for bi in range(active):
                        i = (wb + bi) * VLEN
                        ops.append(("valu", (op1, t1_pool + wave_off + bi * VLEN, val_base + i, c1)))
                    for bi in range(active):
                        i = (wb + bi) * VLEN
                        ops.append(("valu", (op3, t2_pool + wave_off + bi * VLEN, val_base + i, c3)))
                    for bi in range(active):
                        i = (wb + bi) * VLEN
                        ops.append(("valu", (op2, val_base + i, t1_pool + wave_off + bi * VLEN, t2_pool + wave_off + bi * VLEN)))
            for bi in range(active):
                i = (wb + bi) * VLEN
                ops.append(("valu", ("&", t1_pool + wave_off + bi * VLEN, val_base + i, one_vec)))
                ops.append(("valu", ("+", t3_pool + wave_off + bi * VLEN, t1_pool + wave_off + bi * VLEN, one_vec)))
                ops.append(("valu", ("multiply_add", idx_base + i, idx_base + i, two_vec, t3_pool + wave_off + bi * VLEN)))
            for bi in range(active):
                i = (wb + bi) * VLEN
                ops.append(("valu", ("<", t1_pool + wave_off + bi * VLEN, idx_base + i, n_nodes_vec)))
                ops.append(("flow", ("vselect", idx_base + i, t1_pool + wave_off + bi * VLEN, idx_base + i, zero_vec)))
            return ops

        def interleave_ops(a, b, ca=8, cb=8):
            out = []
            ia = ib = 0
            while ia < len(a) or ib < len(b):
                if ia < len(a):
                    out.extend(a[ia : ia + ca])
                    ia += ca
                if ib < len(b):
                    out.extend(b[ib : ib + cb])
                    ib += cb
            return out

        for _ in range(rounds):
            n_windows = (full_blocks + max_inflight_blocks - 1) // max_inflight_blocks
            if n_windows > 0:
                body.extend(emit_window_gather(0, 0))
                for win in range(n_windows):
                    wave = win % wave_count
                    compute_ops = emit_window_compute(win, wave)
                    if win + 1 < n_windows:
                        next_wave = (win + 1) % wave_count
                        next_gather_ops = emit_window_gather(win + 1, next_wave)
                        body.extend(interleave_ops(compute_ops, next_gather_ops, 4, 4))
                    else:
                        body.extend(compute_ops)

            # Tail scalar
            for i in range(tail_start, batch_size):
                body.append(("alu", ("+", tail_t1, self.scratch["forest_values_p"], idx_base + i)))
                body.append(("load", ("load", tail_node, tail_t1)))
                body.append(("alu", ("^", val_base + i, val_base + i, tail_node)))
                for (op1, val1, op2, op3, val3) in HASH_STAGES:
                    body.append(("alu", (op1, tail_t1, val_base + i, self.scratch_const(val1))))
                    body.append(("alu", (op3, tail_t2, val_base + i, self.scratch_const(val3))))
                    body.append(("alu", (op2, val_base + i, tail_t1, tail_t2)))
                body.append(("alu", ("&", tail_t1, val_base + i, one_const)))
                body.append(("alu", ("+", tail_t3, tail_t1, one_const)))
                body.append(("alu", ("*", idx_base + i, idx_base + i, two_const)))
                body.append(("alu", ("+", idx_base + i, idx_base + i, tail_t3)))
                body.append(("alu", ("<", tail_t1, idx_base + i, self.scratch["n_nodes"])))
                body.append(("flow", ("select", idx_base + i, tail_t1, idx_base + i, zero_const)))

        # Final values writeback only (submission harness checks values).
        for b in range(full_blocks):
            i = b * VLEN
            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tail_t1, self.scratch["inp_values_p"], i_const)))
            body.append(("store", ("vstore", tail_t1, val_base + i)))
        for i in range(tail_start, batch_size):
            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tail_t1, self.scratch["inp_values_p"], i_const)))
            body.append(("store", ("store", tail_t1, val_base + i)))

        self.instrs.extend(self.build(body))

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
