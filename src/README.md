# EDT Implementation

Euclidean Distance Transform using uint8 connectivity graphs.


## Algorithm Overview

This implementation uses the separable approach from Meijster et al (2002) and Felzenszwalb & Huttenlocher (2012):

**Key insight**: The N-dimensional EDT can be computed as N sequential 1D transforms, one along each axis. Each pass reads the previous pass's output and updates it.

### Pass 0: 1D Distance Along First Axis

For each 1D scanline along axis 0, compute squared distance to nearest segment boundary:

```
Scanline:    [A][A][A][B][B]     (A and B are different labels)
Pass 0:      [1][4][1][1][4]     (squared distance to boundary)
```

Within a segment, distance grows quadratically from each end: 1, 4, 9, 16...

### Passes 1+: Parabolic Envelope (Felzenszwalb)

For each subsequent axis, we need to combine the previous distances with distances along the new axis. This is done efficiently using the "lower envelope of parabolas" algorithm.

**Intuition**: Each voxel's squared distance from Pass 0 defines a parabola centered at that voxel. The new distance at any position is the minimum across all parabolas - the "lower envelope".

The algorithm:
1. Build a stack of "winning" parabolas from left to right
2. For each new parabola, pop any it dominates
3. Scan right-to-left, evaluating the winning parabola at each position

This runs in O(n) time per scanline.

### Why Segments Matter

The algorithm processes each *segment* (contiguous run of same-label voxels) independently. At segment boundaries, distance resets. This is what makes multi-label EDT work - label 1's distance field doesn't "leak" into label 2.

## Graph-First Architecture

Traditional EDT stores segment labels (uint32) at each voxel, requiring comparisons like `labels[i] != labels[i+1]` to detect boundaries.

**New approach**: Pre-compute (or pass in) connectivity as a uint8 bitfield per voxel. Each bit indicates whether an edge exists to the next voxel along that axis.

```
Traditional:                    Graph-first:
┌───────────────────────┐      ┌───────────────────────┐
│ for each voxel:       │      │ for each voxel:       │
│   if labels[i] !=     │  →   │   if !(graph[i] & 1)  │
│      labels[i+1]      │      │      // boundary      │
│      // boundary      │      │                       │
└───────────────────────┘      └───────────────────────┘
```

**Benefits**:

1. **Memory**: Graph (uint8 for 2D-4D, uint16 for 5D+) vs uint32 labels = 2-4x smaller internal storage
2. **Bandwidth**: Reading 1-2 bytes vs 4 bytes per voxel
3. **Simplicity**: Bit test vs label comparison
4. **Flexibility**: Graph can encode arbitrary boundaries (voxel_graph API)

The graph encodes: "can I continue my segment to the next voxel along this axis?"
- Edge bit set (1): same segment, continue accumulating distance
- Edge bit clear (0): boundary, reset distance computation

## API

```python
import edt

# Standard usage: labels → EDT (graph built internally in C++)
result = edt.edtsq(labels, parallel=8, black_border=True)

# Or build graph explicitly (useful if computing EDT multiple times)
graph = edt.build_graph(labels, parallel=8)
result = edt.edtsq_graph(graph, parallel=8, black_border=True)

# Custom connectivity via voxel_graph (labels optional)
result = edt.edtsq(voxel_graph=custom_graph)
```

## Graph Format

| Property | Value |
|----------|-------|
| Shape | Same as input labels |
| dtype | uint8 (2D-4D), uint16 (5D+) |
| Background | 0 |
| Foreground marker | bit 0 (0b00000001 = 1) |

**Edge bit encoding** (connectivity to next voxel along each axis):

Formula: `bit_position = 2 * (ndim - 1 - axis) + 1`

| Dimension | Axis 0 | Axis 1 | Axis 2 | Axis 3 | dtype |
|-----------|--------|--------|--------|--------|-------|
| 2D | bit 3 (8) | bit 1 (2) | - | - | uint8 |
| 3D | bit 5 (32) | bit 3 (8) | bit 1 (2) | - | uint8 |
| 4D | bit 7 (128) | bit 5 (32) | bit 3 (8) | bit 1 (2) | uint8 |
| 5D-8D | bit 9+ | ... | ... | ... | uint16 |
| 9D-12D | bit 17+ | ... | ... | ... | uint32 |
| 13D-16D | bit 25+ | ... | ... | ... | uint64 |

Note: Bit 0 is reserved for the foreground marker, so 4D is the maximum for uint8.


## Memory Usage

Let N = number of voxels. The graph-first architecture minimizes peak
working-set memory when the caller can release the labels array before
the EDT runs.

- Graph: 1N bytes (uint8) for 2D-4D, 2N bytes (uint16) for 5D+.
- Output: 4N bytes (float32).
- When building from labels: graph is allocated inside C++ and freed
  before the call returns; only the output remains.

| Input label dtype | Graph-first minimum peak | Label-segment peak |
|---|---|---|
| uint8  | 5N (1N graph + 4N output) | 5N (1N labels + 4N output) |
| uint16 | 5N | 6N |
| uint32 | 5N | 8N |

**Caveat (`edtsq(labels)` fused path)**: the Python labels array remains
held by the caller for the duration of the call, so the actual peak is
the labels-array size plus the graph (1-2N) plus the output (4N) -- i.e.
roughly (K + 5)N where K = label dtype bytes. To realize the minimum
peak above, use the explicit two-step pattern:

```python
graph = edt.build_graph(labels, parallel=...)
del labels  # release before EDT
dist = edt.edtsq_graph(graph, parallel=...)
```

The label-segment approach does not benefit from this pattern because
its algorithm requires the labels array throughout.

### Voxel Graph Input

When using `voxel_graph` input, the bidirectional cc3d format is translated to the internal ND graph format:

1. **Mask out negative direction bits** - voxel_graph uses 2 bits per axis (positive + negative); ND graph uses only forward edges
2. **Add foreground marker** - bit 0 (0b00000001) is set for non-zero voxels

This creates a temporary uint8 array (1N bytes), but avoids the grid doubling required by the legacy label-segment approach. Assuming 16-bit labels (2N memory allocation):

**Peak memory for voxel_graph input**:

| Component | Graph-first | Legacy 2D | Legacy 3D |
|-----------|-------------|-----------|-----------|
| Input voxel_graph | 1N | 1N | 1N |
| Translated ND graph | 1N | - | - |
| Labels (legacy requires) | - | 2N | 2N |
| double_labels (uint8) | - | 4N | 8N |
| Transform on doubled grid | - | 16N | 32N |
| Output (float32) | 4N | 4N | 4N | 
| **Peak** | **6N** | **23N** | **43N** |
| **Theoretical savings** | 1x | 3.8x | 7.2x |

## Implementation Details

### Segment Detection

Both graph-first and label-segment approaches use fused segment detection - boundaries are detected during the EDT passes rather than in a separate labeling step.

The difference is how boundaries are detected:

- **Graph-first**: Check edge bits (`if !(graph[i] & bit)`)
- **Label-segment**: Compare adjacent labels (`if labels[i] != labels[i+1]`)

The graph approach uses less memory bandwidth (1-2 bytes vs 4 bytes per voxel for uint32 labels).

### Threading

Each pass is parallelized across independent scanlines using a shared thread pool.

This implementation adds work-based capping to avoid thread overhead on small arrays:
- < 60K voxels: max 4 threads
- < 120K voxels: max 8 threads
- < 400K voxels: max 12 threads

### Axis Processing Order

Both implementations process the innermost axis (stride=1) first, then work outward to larger strides. This ensures sequential memory access patterns in the cache-critical first pass.

## References

- Meijster, A., Roerdink, J.B.T.M., Hesselink, W.H. (2002). "A General Algorithm for Computing Distance Transforms in Linear Time"
- Felzenszwalb, P.F., Huttenlocher, D.P. (2012). "Distance Transforms of Sampled Functions"
- Saito, T., Toriwaki, J. (1994). "New algorithms for Euclidean distance transformation"
