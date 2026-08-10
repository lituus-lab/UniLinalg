# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## C ABI for UniLinalg. Built --app:staticlib/--app:lib --noMain --mm:arc -d:danger.
## Keep in sync with include/UniLinalg.h; tests/c links the header against this lib.
##
## Handle-based for Matrix/CsrMatrix (fixed to float64 for the C surface):
## pinned `ref Matrix[float64]` / `ref CsrMatrix[float64]` owned by the C host
## until the matching `*_destroy`. The C host MUST call `ulin_init()` once
## before any other entry point (same doctrine as unimath_init: brings up the
## Nim/ARC runtime under `--app:lib`). Never raises: returns NULL / a negative
## count / a ULIN_ERR_* code on bad input or domain errors, never propagates a
## Nim exception across the boundary.
##
## Vector[2/3/4, float64] is value-type: `D` is a compile-time constant per
## alias, so flat `ulin_vec2/3/4` structs are the natural mapping -- no handle,
## no heap allocation, matching the family's "handle for dynamically-sized,
## struct for statically-sized" convention (see the book).
import ../UniLinalg

const UniLinalgVersionC: cstring = "1.0.0"

const
  ULIN_OK = cint(0)
  ULIN_ERR_NULL_HANDLE = cint(1)
  ULIN_ERR_SHAPE_MISMATCH = cint(2)
  ULIN_ERR_SINGULAR = cint(3)
  ULIN_ERR_NOT_SPD = cint(4)
  ULIN_ERR_BUFFER_TOO_SMALL = cint(5)

  # rows*cols must fit in the cint element counts ulin_matrix_get_buffer
  # (and rows/cols themselves) return -- above this, `cint(n)` silently wraps
  # to a wrong, possibly negative count under -d:danger (range checks
  # compiled away; confirmed by direct testing, not just reasoned about).
  MaxElemCount = int(high(int32))

# ------------------------------------------------------------------------------
# Internal helpers (NOT exported): pinned handles, buffer <-> seq, error codes.
# ------------------------------------------------------------------------------

type AbiMatrix = ref Matrix[float64]

proc pin(m: Matrix[float64]): pointer =
  let r = new(AbiMatrix)
  r[] = m
  GC_ref(r) # pin beyond ARC; the C host now owns the reference
  cast[pointer](r)

proc matOf(h: pointer): Matrix[float64] {.inline.} =
  if h == nil: return initMatrix[float64](1, 1)
  cast[AbiMatrix](h)[]

proc unrefMatrix(h: pointer) {.inline.} =
  if h != nil: GC_unref(cast[AbiMatrix](h))

type AbiSparse = ref CsrMatrix[float64]

proc pinSparse(m: CsrMatrix[float64]): pointer =
  let r = new(AbiSparse)
  r[] = m
  GC_ref(r)
  cast[pointer](r)

proc sparseOf(h: pointer): CsrMatrix[float64] {.inline.} =
  if h == nil: return toCsr(initMatrix[float64](1, 1))
  cast[AbiSparse](h)[]

proc unrefSparse(h: pointer) {.inline.} =
  if h != nil: GC_unref(cast[AbiSparse](h))

proc ptrToSeq(p: ptr float64, n: csize_t): seq[float64] =
  ## Copies `n` float64s starting at `p` into a fresh seq. Empty seq if
  ## `p` is nil or `n` is 0.
  result = newSeq[float64](int(n))
  if p != nil and n > 0:
    copyMem(addr result[0], p, int(n) * sizeof(float64))

proc seqToBuf(s: seq[float64], buf: ptr float64) {.inline.} =
  if buf != nil and s.len > 0:
    copyMem(buf, unsafeAddr s[0], s.len * sizeof(float64))

# Internal init-once flag. Declared outside the exportc block (unimath_init's
# doctrine): a writable bool exported as a raw C symbol would let a host
# bypass ulin_init().
var gInited: bool

# Unmangled C symbols, C calling convention, exported from the shared lib.
{.push exportc, cdecl, dynlib.}

# ------------------------------------------------------------------------------
# Version & lifecycle.
# ------------------------------------------------------------------------------

proc ulin_version(): cstring =
  ## Static version string; do not free.
  UniLinalgVersionC

proc NimMain() {.importc, cdecl.}

proc ulin_init(): bool =
  ## Bring up the Nim/ARC runtime. Call once before any other entry point.
  ## Idempotent. Returns true.
  if not gInited:
    NimMain()
    gInited = true
  true

proc ulin_cleanup() =
  ## No-op (matches ulin_init); handles are freed per-call by `*_destroy`.
  discard

proc ulin_get_error_string(error_code: cint): cstring =
  case int(error_code)
  of 0: "Success"
  of 1: "Null handle"
  of 2: "Shape mismatch"
  of 3: "Singular matrix"
  of 4: "Matrix is not symmetric positive-definite"
  of 5: "Output buffer too small"
  else: "Unknown error"

# ------------------------------------------------------------------------------
# Matrix — handle = pinned ref Matrix[float64].
# ------------------------------------------------------------------------------

proc ulin_matrix_create(rows, cols: cint): pointer =
  ## Zero matrix of the given shape. NULL if rows/cols <= 0 or rows*cols
  ## overflows what a cint element count can represent.
  if rows <= 0 or cols <= 0: return nil
  if int(rows) * int(cols) > MaxElemCount: return nil
  pin(initMatrix[float64](int(rows), int(cols)))

proc ulin_matrix_destroy(h: pointer) =
  unrefMatrix(h)

proc ulin_matrix_rows(h: pointer): cint =
  ## O(1): reads the field through the ref, no Matrix (and its seq) copy.
  if h == nil: return 0
  cint(cast[AbiMatrix](h)[].rows)

proc ulin_matrix_cols(h: pointer): cint =
  if h == nil: return 0
  cint(cast[AbiMatrix](h)[].cols)

proc ulin_matrix_get(h: pointer, i, j: cint): float64 =
  ## 0.0 on a nil handle or out-of-range index (never raises). Reads through
  ## the ref (no Matrix/seq copy) -- O(1), not O(rows*cols).
  if h == nil: return 0.0
  let m = cast[AbiMatrix](h)
  if i < 0 or j < 0 or int(i) >= m[].rows or int(j) >= m[].cols: return 0.0
  m[][int(i), int(j)]

proc ulin_matrix_set(h: pointer, i, j: cint, v: float64) =
  ## No-op on a nil handle or out-of-range index.
  if h == nil: return
  var m = cast[AbiMatrix](h)
  if i < 0 or j < 0 or int(i) >= m[].rows or int(j) >= m[].cols: return
  m[][int(i), int(j)] = v

proc ulin_matrix_create_from_buffer(rows, cols: cint, buf: ptr float64,
                                     n: csize_t): pointer =
  ## Matrix from a flat row-major buffer (element (i,j) at buf[i*cols+j]) --
  ## one bulk copy instead of rows*cols individual `ulin_matrix_set` calls.
  ## NULL if rows/cols <= 0, buf is nil, n != rows*cols, or rows*cols
  ## overflows what a cint element count can represent.
  if rows <= 0 or cols <= 0 or buf == nil: return nil
  if int(rows) * int(cols) > MaxElemCount: return nil
  if int(n) != int(rows) * int(cols): return nil
  pin(matrix[float64](int(rows), int(cols), ptrToSeq(buf, n)))

proc ulin_matrix_get_buffer(h: pointer, outBuf: ptr float64,
                             outCap: csize_t): cint =
  ## Bulk row-major read of every element into outBuf -- one copy instead of
  ## rows*cols individual `ulin_matrix_get` calls. Returns the count written
  ## (rows*cols), or the negated ULIN_ERR_* reason (-ULIN_ERR_NULL_HANDLE /
  ## -ULIN_ERR_BUFFER_TOO_SMALL) on failure -- always negative, so an
  ## existing `< 0` check keeps working unchanged. rows*cols too large for a
  ## cint count (e.g. via ulin_matrix_mul: two individually in-bounds
  ## operands can still multiply out past MaxElemCount) is also reported as
  ## -ULIN_ERR_BUFFER_TOO_SMALL: no outCap could satisfy it either way.
  if h == nil or outBuf == nil: return -ULIN_ERR_NULL_HANDLE
  let m = cast[AbiMatrix](h)
  let n = m[].rows * m[].cols
  if n > MaxElemCount or int(outCap) < n: return -ULIN_ERR_BUFFER_TOO_SMALL
  seqToBuf(m[].data, outBuf)
  cint(n)

proc ulin_matrix_add(a, b: pointer): pointer =
  ## NULL on a nil handle or shape mismatch.
  if a == nil or b == nil: return nil
  let ma = matOf(a)
  let mb = matOf(b)
  if ma.rows != mb.rows or ma.cols != mb.cols: return nil
  pin(ma + mb)

proc ulin_matrix_sub(a, b: pointer): pointer =
  if a == nil or b == nil: return nil
  let ma = matOf(a)
  let mb = matOf(b)
  if ma.rows != mb.rows or ma.cols != mb.cols: return nil
  pin(ma - mb)

proc ulin_matrix_mul(a, b: pointer): pointer =
  ## NULL on a nil handle or `a.cols != b.rows`.
  if a == nil or b == nil: return nil
  let ma = matOf(a)
  let mb = matOf(b)
  if ma.cols != mb.rows: return nil
  pin(ma * mb)

proc ulin_matrix_scale(h: pointer, s: float64): pointer =
  if h == nil: return nil
  pin(s * matOf(h))

proc ulin_matrix_transpose(h: pointer): pointer =
  if h == nil: return nil
  pin(transpose(matOf(h)))

proc ulin_matrix_almost_equal(a, b: pointer, eps: float64): cint =
  ## 0 (false) if either handle is nil.
  if a == nil or b == nil: return cint(0)
  cint(almostEqual(matOf(a), matOf(b), eps))

proc ulin_matrix_determinant(h: pointer, out_ok: ptr cint): float64 =
  ## 0.0 with `*out_ok = false` on a nil handle or a non-square matrix.
  if h == nil:
    if out_ok != nil: out_ok[] = cint(false)
    return 0.0
  let m = matOf(h)
  if not m.isSquare:
    if out_ok != nil: out_ok[] = cint(false)
    return 0.0
  if out_ok != nil: out_ok[] = cint(true)
  det(m)

proc ulin_matrix_lu_solve(h: pointer, b: ptr float64, blen: csize_t,
                          outBuf: ptr float64, outCap: csize_t,
                          refine: bool): cint =
  ## Solves Ax = b, writing x into outBuf (outCap must be >= rows). `refine`
  ## true runs one step of UniAccurate-backed iterative refinement after the
  ## solve, correcting the 1-2 ULP a plain float64 solve can miss (ADR-0006).
  ## Returns the number of elements written, or the negated ULIN_ERR_* reason
  ## (-ULIN_ERR_NULL_HANDLE / -ULIN_ERR_SHAPE_MISMATCH /
  ## -ULIN_ERR_BUFFER_TOO_SMALL / -ULIN_ERR_SINGULAR) on failure -- always
  ## negative, so an existing `< 0` failure check keeps working unchanged.
  if h == nil or b == nil or outBuf == nil: return -ULIN_ERR_NULL_HANDLE
  let m = matOf(h)
  if not m.isSquare or int(blen) != m.rows: return -ULIN_ERR_SHAPE_MISMATCH
  if int(outCap) < m.rows: return -ULIN_ERR_BUFFER_TOO_SMALL
  let x =
    try: solve(m, ptrToSeq(b, blen), useRefinement = refine)
    except ValueError: return -ULIN_ERR_SINGULAR
  seqToBuf(x, outBuf)
  cint(x.len)

proc ulin_matrix_cholesky(h: pointer): pointer =
  ## Lower-triangular L with A = L L^T. NULL on a nil handle, a non-square
  ## matrix, or a matrix that is not symmetric positive-definite.
  ##
  ## The non-square check is required here, not just trusted to cholesky()'s
  ## own `require:` -- that guard is debug-only (compiled away under
  ## -d:danger, this build's own flag) and confirmed by direct testing to
  ## silently return a wrong-but-plausible-looking result on non-square
  ## input otherwise (the algorithm reads only the first `rows` columns,
  ## silently ignoring the rest).
  if h == nil: return nil
  let m = matOf(h)
  if not m.isSquare: return nil
  try: pin(cholesky(m))
  except ValueError: nil

proc ulin_matrix_qr(h: pointer, out_q, out_r: ptr pointer): cint =
  ## Householder QR: A = Q R. Writes the Q and R handles through `out_q`/
  ## `out_r`. Returns ULIN_OK, or ULIN_ERR_NULL_HANDLE / ULIN_ERR_SHAPE_MISMATCH
  ## (rows < cols) on failure -- neither out-param is written on error.
  if h == nil or out_q == nil or out_r == nil: return ULIN_ERR_NULL_HANDLE
  let m = matOf(h)
  if m.rows < m.cols: return ULIN_ERR_SHAPE_MISMATCH
  let d = qrDecompose(m)
  out_q[] = pin(d.q)
  out_r[] = pin(d.r)
  ULIN_OK

proc ulin_matrix_svd(h: pointer, out_u: ptr pointer,
                      out_s: ptr float64, out_s_cap: csize_t,
                      out_v: ptr pointer): cint =
  ## One-sided Jacobi SVD: A = U diag(S) V^T. Writes U/V handles and the
  ## singular values (descending, `cols` of them) into the caller's buffers.
  ## Returns ULIN_OK, ULIN_ERR_NULL_HANDLE, ULIN_ERR_SHAPE_MISMATCH (rows <
  ## cols), or ULIN_ERR_BUFFER_TOO_SMALL. No out-param is written on error.
  if h == nil or out_u == nil or out_s == nil or out_v == nil:
    return ULIN_ERR_NULL_HANDLE
  let m = matOf(h)
  if m.rows < m.cols: return ULIN_ERR_SHAPE_MISMATCH
  if int(out_s_cap) < m.cols: return ULIN_ERR_BUFFER_TOO_SMALL
  let d = svdDecompose(m)
  out_u[] = pin(d.u)
  out_v[] = pin(d.v)
  seqToBuf(d.s, out_s)
  ULIN_OK

# ------------------------------------------------------------------------------
# Sparse — handle = pinned ref CsrMatrix[float64].
# ------------------------------------------------------------------------------

proc ulin_sparse_from_dense(h: pointer): pointer =
  if h == nil: return nil
  pinSparse(toCsr(matOf(h)))

proc ulin_sparse_to_dense(h: pointer): pointer =
  if h == nil: return nil
  pin(toDense(sparseOf(h)))

proc ulin_sparse_destroy(h: pointer) =
  unrefSparse(h)

proc ulin_sparse_nnz(h: pointer): cint =
  if h == nil: return 0
  cint(sparseOf(h).nnz)

proc ulin_sparse_matvec(h: pointer, v: ptr float64, vlen: csize_t,
                         outBuf: ptr float64, outCap: csize_t): cint =
  ## Sparse matrix-vector product. Returns the number of elements written,
  ## or -1 on a nil handle/buffer, shape mismatch, or too-small buffer.
  if h == nil or v == nil or outBuf == nil: return cint(-1)
  let m = sparseOf(h)
  if int(vlen) != m.cols: return cint(-1)
  if int(outCap) < m.rows: return cint(-1)
  let x = m * ptrToSeq(v, vlen)
  seqToBuf(x, outBuf)
  cint(x.len)

# ------------------------------------------------------------------------------
# Vector2/3/4 — value types (float64 only; D is a compile-time constant per
# alias, so no handle is needed).
# ------------------------------------------------------------------------------

type
  ulin_vec2 {.bycopy, exportc: "ulin_vec2".} = object
    x*, y*: float64
  ulin_vec3 {.bycopy, exportc: "ulin_vec3".} = object
    x*, y*, z*: float64
  ulin_vec4 {.bycopy, exportc: "ulin_vec4".} = object
    x*, y*, z*, w*: float64

proc ulin_vec2_add(a, b: ulin_vec2): ulin_vec2 =
  let r = vec2(a.x, a.y) + vec2(b.x, b.y)
  ulin_vec2(x: r.x, y: r.y)

proc ulin_vec2_sub(a, b: ulin_vec2): ulin_vec2 =
  let r = vec2(a.x, a.y) - vec2(b.x, b.y)
  ulin_vec2(x: r.x, y: r.y)

proc ulin_vec2_scale(a: ulin_vec2, s: float64): ulin_vec2 =
  let r = vec2(a.x, a.y) * s
  ulin_vec2(x: r.x, y: r.y)

proc ulin_vec2_dot(a, b: ulin_vec2): float64 =
  dot(vec2(a.x, a.y), vec2(b.x, b.y))

proc ulin_vec2_length(a: ulin_vec2): float64 =
  vec2(a.x, a.y).length

proc ulin_vec2_normalize(a: ulin_vec2): ulin_vec2 =
  let r = vec2(a.x, a.y).normalize
  ulin_vec2(x: r.x, y: r.y)

proc ulin_vec2_almost_equal(a, b: ulin_vec2, eps: float64): cint =
  cint(almostEqual(vec2(a.x, a.y), vec2(b.x, b.y), eps))

proc ulin_vec2_cross2d(a, b: ulin_vec2): float64 =
  cross2d(vec2(a.x, a.y), vec2(b.x, b.y))

proc ulin_vec2_perp(a: ulin_vec2): ulin_vec2 =
  let r = perp(vec2(a.x, a.y))
  ulin_vec2(x: r.x, y: r.y)

proc ulin_vec2_perp_cw(a: ulin_vec2): ulin_vec2 =
  let r = perpCW(vec2(a.x, a.y))
  ulin_vec2(x: r.x, y: r.y)

proc ulin_vec3_add(a, b: ulin_vec3): ulin_vec3 =
  let r = vec3(a.x, a.y, a.z) + vec3(b.x, b.y, b.z)
  ulin_vec3(x: r.x, y: r.y, z: r.z)

proc ulin_vec3_sub(a, b: ulin_vec3): ulin_vec3 =
  let r = vec3(a.x, a.y, a.z) - vec3(b.x, b.y, b.z)
  ulin_vec3(x: r.x, y: r.y, z: r.z)

proc ulin_vec3_scale(a: ulin_vec3, s: float64): ulin_vec3 =
  let r = vec3(a.x, a.y, a.z) * s
  ulin_vec3(x: r.x, y: r.y, z: r.z)

proc ulin_vec3_dot(a, b: ulin_vec3): float64 =
  dot(vec3(a.x, a.y, a.z), vec3(b.x, b.y, b.z))

proc ulin_vec3_length(a: ulin_vec3): float64 =
  vec3(a.x, a.y, a.z).length

proc ulin_vec3_normalize(a: ulin_vec3): ulin_vec3 =
  let r = vec3(a.x, a.y, a.z).normalize
  ulin_vec3(x: r.x, y: r.y, z: r.z)

proc ulin_vec3_almost_equal(a, b: ulin_vec3, eps: float64): cint =
  cint(almostEqual(vec3(a.x, a.y, a.z), vec3(b.x, b.y, b.z), eps))

proc ulin_vec3_cross(a, b: ulin_vec3): ulin_vec3 =
  let r = cross(vec3(a.x, a.y, a.z), vec3(b.x, b.y, b.z))
  ulin_vec3(x: r.x, y: r.y, z: r.z)

proc ulin_vec4_add(a, b: ulin_vec4): ulin_vec4 =
  let r = vec4(a.x, a.y, a.z, a.w) + vec4(b.x, b.y, b.z, b.w)
  ulin_vec4(x: r.x, y: r.y, z: r.z, w: r.w)

proc ulin_vec4_sub(a, b: ulin_vec4): ulin_vec4 =
  let r = vec4(a.x, a.y, a.z, a.w) - vec4(b.x, b.y, b.z, b.w)
  ulin_vec4(x: r.x, y: r.y, z: r.z, w: r.w)

proc ulin_vec4_scale(a: ulin_vec4, s: float64): ulin_vec4 =
  let r = vec4(a.x, a.y, a.z, a.w) * s
  ulin_vec4(x: r.x, y: r.y, z: r.z, w: r.w)

proc ulin_vec4_dot(a, b: ulin_vec4): float64 =
  dot(vec4(a.x, a.y, a.z, a.w), vec4(b.x, b.y, b.z, b.w))

proc ulin_vec4_length(a: ulin_vec4): float64 =
  vec4(a.x, a.y, a.z, a.w).length

proc ulin_vec4_normalize(a: ulin_vec4): ulin_vec4 =
  let r = vec4(a.x, a.y, a.z, a.w).normalize
  ulin_vec4(x: r.x, y: r.y, z: r.z, w: r.w)

{.pop.}
