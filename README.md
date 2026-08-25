# GEMM

Matrix multiplication, optimized from scratch in C on Apple Silicon. Single core.
Went from naive to 111 GFLOPS, which is about 86% of the theoretical peak on
this machine.

I started this because of a different project. I wrote
[Echo](https://github.com/monsihv/Echo), an autodiff engine and deep learning
framework in C, and once I moved it off `malloc`/`free` onto arenas the bottleneck got
really obvious. The whole thing was matmul. Everything else was noise. So I put the
library down and came here to make matmul fast.

This README walks the ladder in the order I actually did it, including the two rungs
where I was wrong about what was happening. I left those in on purpose. The wrong turns
are the part I learned the most from.


## The machine

Everything here is on an M3 Pro, and that matters more than usual:

- 64KB L1 data cache, 16MB shared L2
- 128 byte cache lines, not the 64 bytes basically all the literature assumes
- ARM NEON, 32 vector registers, 128 bits each
- Really high unified memory bandwidth
- ~4 GHz single core

All the conclusions in here are conclusions about this chip. A few of them would
come out different on x86 and I'll say so where it matters.

### Where the theoretical peak comes from

Took me two tries to derive this so I'm writing it down:

```
clock  x  FMA units  x  lanes per vector  x  flops per FMA
~4e9    x      4     x         4          x       2       =  ~128 GFLOPS
```

An FMA counts as 2 flops because it's a multiply and an add fused together. The M3 Pro
P core has 4 NEON pipes that can each issue one FMA per cycle. And a `float32x4_t` is 4
floats, so every one of those FMAs is doing 4 lanes at once. Miss any of those factors
and you get 32 or 64 and think you're way closer to the ceiling than you are.

~128 GFLOPS is the number everything below gets measured against.

## Building

```sh
make run      # -O2 -mcpu=native, runs the sweep, writes results.csv
make debug    # -O0 -g -fsanitize=address
```

Unity build. `src/main.c` includes the .c files directly.


## The ladder

### 1. Naive (ijk)

This is the "mathematical" way to think about matmul. Every output element is one dot
product of a row and a column, so that's how I wrote it.

It's also the worst possible way to walk memory. Going down a column of B means every
access pulls a whole 128 byte cache line to use 4 bytes of it. It never even clears
2.5 GFLOPS, and it falls apart completely at big N, 0.63 at N=4096.

### 2. Loop reorder (ikj)

I started reading and watching stuff about cache locality and strides, and derived in
my notebook that ijk and ikj are the exact same math. The difference is that instead of
finishing one dot product at a time, ikj computes one term of every dot product
across a whole row of the output. Same flops, same answer, but now the inner loop walks
B and C contiguously.

~29 GFLOPS, and dead flat from N=320 out to N=5120. No rolloff at all, even when the
matrices are ~100MB each.

Two things I figured out later:

- The access pattern win on its own is about 3-4x. That part I predicted.
- The rest, going from around 6 up to around 28, was the compiler auto vectorizing the reordered loop. At the
  time I thought I earned all of it. I did not.

So ikj became the baseline to beat. ~29 against a ~128 ceiling is a huge gap.

### 3. Cache blocking, or: a null result

Tiling is the first thing everybody reaches for so I reached for it. The idea is
temporal locality. Don't fill your cache with new data and then have to go back to RAM
for something you were just using, because RAM is orders of magnitude slower.

I finally went and actually measured my cache geometry instead of assuming it (this is
also around when I did a real deep dive into ARM vs x86 and finally learned what
registers and cache actually *are*). Set all three tiles to the same size, kept ikj on
the inside.

It was worse. 24 against ikj's 29.

I was genuinely confused. This is the first optimization everyone shows for GEMM. But
nobody ever shows code, and tiling is easy to mess up, so I spent a full day going back
through it making sure it did what I thought it did. It did. Still lost.

Once I was sure the code did what I meant, the conclusion had to be that tiling just
wasn't buying anything here. My theory at the time was unified memory. I hit the
ballpark but missed the target.

Here's what I think is actually going on, which I only got to later. This machine has a
huge amount of memory bandwidth, and matmul's access pattern is about as predictable as
it gets. My prefetcher was putting in overtime and pulling data into cache before I even
asked for it, so tiling had nothing left to do. All the material I was reading is x86
oriented, where that tradeoff looks very different.

The extra three nested loops aren't free either. You pay real loop overhead for tiling
even when the total inner iteration count is identical.

Wrote it up as a null result and moved on.

### 4. Register blocking

If memory wasn't the constraint, next place to look is the pipeline.

First attempt was 1D, four accumulators along j. Way worse than ikj.
Diagnosing that taught me more than fixing it did: five loads feeding four FMAs is just
a baby loop reorder. I was pulling a 128 byte line to use 16 bytes of it.

The 2D version is a 4x4 tile, 16 scalar accumulators, 4 values of A held as constants,
each one reused across four columns of B. ~26 GFLOPS, about 3x the 1D version.

The stuff that made this click, in the order it mattered:

- Dependency chains. One accumulator serializes on FMA latency. My kernel was
  latency bound, sitting there waiting on a result when the unit could have started
  another one. Breaking the chain is most of the win.
- Out of order execution and data hazards. Multiple accumulators break the true
  dependency. Register renaming handles the false ones for free.
- Keeping things in registers. Those 16 accumulators have to stay unrolled scalars.
  Write them as `sum[m][n]` in a nested loop and they land in memory and you've undone
  the whole thing. This is exactly why real BLAS ships hand written microkernels per
  architecture.

### 5. Blocking comes back from the dead

Now that I had a 4x4 tile in flight I was touching a lot of B at once, so I figured
maybe tiling is useful again after all, keep that big block in L1. Tried it. Small
improvement, nothing dramatic.

Then I ran the sweep past N=2048 and the register kernel fell off a cliff:
25 → 18.6 → 8.3 → 4.5.

The microkernel walks a 4 wide column strip of B across N distinct cache lines. Past a
certain N that strip stops fitting in L1 and every single k iteration re-fetches it.

So blocking was never wrong. It was conditional on the kernel. ikj had nothing to
block. The register microkernel *manufactures* the exact working set that tiling exists
to fix. Re-fusing them brought the top end back and took N=4096 from 4.5 to ~12.

Found out afterwards this is also why every tutorial does register blocking before
tiling. Nobody mentions that the order is load bearing.

### 6. Doing the vectorization myself (NEON)

The thing that finally explained the gap was compiling everything at `-O0`.

Everything tanked except one kernel. That's when it clicked that ikj's 29 was mostly the
auto vectorizer, and that my register blocked kernel wasn't being vectorized at all -
clang's cost model kept declining it. So my ~26 was pure scalar work going up against a
vectorized ikj with four lanes sitting completely idle.

I was compute bound with the compute unused. There were NEON registers just sitting
there waiting for me.

First pass with `arm_neon.h` intrinsics (`vld1q_f32`, `vfmaq_n_f32`, `vst1q_f32`) got
me to ~35 GFLOPS.

Then I did the same trick one level up. If multiple scalar accumulators broke the
dependency chain, multiple vector accumulators should too. So treat each vector like
a scalar and unroll again. 4 rows of A x 4 vectors of B = 16 vector accumulators,
64 output floats live at once, `j += 16`.

~80-90 GFLOPS. Then I flipped the i/j loop order and retiled to iTile=64,
kTile=jTile=128 and hit a peak of 111 GFLOPS at N=832.

#### Why the tile is exactly 4 rows x 16 columns

This is a two sided constraint and I want to write down both sides.

You have 32 vector registers, 4 floats each. Count what's live every k iteration:

| what | registers |
|---|---:|
| C accumulators (4 rows x 16 cols = 64 floats) | 16 |
| B strip (16 contiguous floats of row k) | 4 |
| A constants (scalars alias the same register file on AArch64) | 4 |
| **total** | **24 of 32** |

Push it any bigger and you spill. `j += 32` needs 32 accumulators on its own, the whole
file, before B and A even show up. Same for 8 rows x 4 vectors. And a spilled
accumulator is a memory access in your innermost loop, which is the exact thing this
optimization exists to prevent.

Other side is arithmetic intensity, or how much work you get per byte you load. Per k
iteration this kernel loads 20 floats (4 from A, 16 from B) and does 16 vector FMAs =
128 flops. That's 6.4 flops per float loaded. The 1D version was 5 floats for 8
flops, so 1.6. Four times worse, which is basically the performance gap I measured.

So: bigger tile for more intensity, stop right before you spill. 4x16 is where those
two meet on this chip.


## Results

`results.csv`, N = 64 to 5120, all correctness checked against ikj with an epsilon
tolerance. (Floating point addition isn't associative, so once k splits across multiple
tiles the summation order changes and exact `!=` comparison is the wrong test. Took me
a minute to figure out that one was the checker's fault and not the kernel's.)

What the columns actually are:

| column | what it is |
|---|---|
| **Naive** | ijk, textbook dot product order |
| **Cache** | ikj loop reorder, the baseline to beat |
| **Blocking** | cache blocking + register blocking fused, scalar |
| **SIMD** | same thing with a hand written NEON microkernel |

GFLOPS:

| N | Naive | Cache (ikj) | Blocking | SIMD |
|---:|---:|---:|---:|---:|
| 64 | 0.84 | 5.24 | 4.85 | 17.48 |
| 128 | 0.74 | 8.08 | 6.36 | 22.19 |
| 192 | 1.15 | 13.25 | 11.67 | 42.77 |
| 256 | 1.49 | 21.63 | 19.65 | 69.91 |
| 320 | 2.11 | 28.59 | 25.36 | 103.53 |
| 384 | 2.47 | 28.58 | 26.50 | 105.44 |
| 448 | 2.46 | 28.85 | 26.81 | 106.22 |
| 512 | 2.42 | 28.94 | 27.87 | 75.94 |
| 576 | 2.40 | 28.77 | 25.97 | 109.23 |
| 640 | 2.38 | 28.65 | 25.29 | 109.00 |
| 704 | 2.36 | 28.67 | 25.51 | 109.76 |
| 768 | 2.34 | 29.01 | 24.89 | 101.84 |
| 832 | 2.30 | 29.69 | 24.63 | **111.12** |
| 896 | 2.31 | 28.87 | 23.75 | 110.55 |
| 960 | 2.26 | 28.44 | 23.10 | 109.46 |
| 1024 | 1.96 | 28.89 | 24.69 | 67.09 |
| 2048 | 1.84 | 27.50 | 19.94 | 56.99 |
| 3072 | 1.21 | 28.98 | 18.46 | 70.45 |
| 4096 | 0.63 | 28.75 | 10.89 | 54.46 |
| 5120 | 0.73 | 28.74 | 20.32 | 67.45 |

111 GFLOPS out of ~128 theoretical is about 86%.


## Stuff I learned that wasn't the point

- Bandwidth is not latency. This machine has a ton of unified memory bandwidth, and
  matmul's access pattern is about as prefetcher friendly as it gets. My prefetcher was
  putting in overtime and getting data into cache before I needed it, which is a real
  reason tiling underdelivers here compared to all the x86 material.
- L1 isn't the only budget. My first tiling attempt sized everything against L1.
  Reworking it per operand and sizing the long reuse operand against the 16MB L2 is what
  finally made the big N cases behave.
- Measure your own machine. 128 byte lines, real `sysctl` numbers, not whatever the
  textbook says.
- A null result is a result as long as you can explain why it happened.

## Known limitations and open questions

- Power of two dips. SIMD drops hard at N = 512, 1024, 2048, 4096 compared to its
  neighbors, 75.9 at 512 when 448 and 576 are at 106 and 109. Non powers of two like
  3072 and 5120 don't do it. Haven't solved this yet.
- No ragged edge handling in the SIMD microkernel. Every benchmark size is a multiple
  of 16. Real BLAS ships separate cleanup kernels, I haven't written one.
- Single core only.
- CPU only. A GPU compute backend is an eventual benchmark, not part of this repo.
- There's duplication between the scalar register blocking kernel and its tiled helper.
  I know. It's on the list of things to abstract.

## References

CSAPP ch5, Drepper's *What Every Programmer Should Know About Memory*, the
how-to-optimize-gemm wiki, and a bunch of CppCon talks. All of it conceptual, since the
literature is x86 oriented, so every actual number in here had to be re-derived for ARM.


## A note on this README

The content here is mine. The reasoning, the code, the wrong turns, the derivations, and the
conclusions all came out of my own notes and my own work on this repo. I used an AI
assistant to help structure my words and to format and draft the markdown. I also used it
to help me with the makefile.
