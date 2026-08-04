# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
## Splice the bench markdown fragment `nimble benchReadme` wrote
## (bench/.md_bench.md, gitignored) into bench/README.md's Benchmarks
## section, tagged to the machine that ran it.
##
## `<!-- bench:machine=<slug> -->` ... `<!-- /bench:machine=<slug> -->`
## wraps the block. Re-running on the same machine replaces only that
## block; a different slug (or the `UNILINALG_BENCH_MACHINE` env var, for
## a box whose auto-detected slug is not the one you want recorded, e.g.
## a FreeBSD/Zen4 box) appends alongside it -- one README can carry
## results from several machines without one overwriting another's.
import std/[os, osproc, strutils]

proc machineSlug(): string =
  if existsEnv("UNILINALG_BENCH_MACHINE"):
    return getEnv("UNILINALG_BENCH_MACHINE")
  var cpu = hostCPU
  when defined(macosx):
    let (brand, code) = execCmdEx("sysctl -n machdep.cpu.brand_string")
    if code == 0: cpu = brand.strip()
  elif defined(linux):
    let (model, code) = execCmdEx("sh -c \"grep -m1 'model name' /proc/cpuinfo | cut -d: -f2\"")
    if code == 0 and model.strip().len > 0: cpu = model.strip()
  result = (hostOS & "-" & cpu).toLowerAscii().multiReplace(
    (" ", "-"), ("(", ""), (")", ""), ("_", "-"))
  while "--" in result: result = result.replace("--", "-")

proc spliceReadme(path: string, slug: string, body: string) =
  let content = readFile(path)
  let startTag = "<!-- bench:machine=" & slug & " -->"
  let endTag = "<!-- /bench:machine=" & slug & " -->"
  let full = startTag & "\n" & body & "\n" & endTag
  if startTag in content:
    let s = content.find(startTag)
    let e = content.find(endTag) + endTag.len
    writeFile(path, content[0 ..< s] & full & content[e .. ^1])
    stderr.writeLine("[readme] replaced block for " & slug)
  else:
    const marker = "<!-- bench:insert -->"
    if marker notin content:
      stderr.writeLine("[readme] no <!-- bench:insert --> marker in " & path & ", skip splice")
      return
    let idx = content.find(marker) + marker.len
    writeFile(path, content[0 ..< idx] & "\n\n" & full & content[idx .. ^1])
    stderr.writeLine("[readme] inserted block for " & slug)

proc main() =
  const fragPath = "bench/.md_bench.md"
  const readmePath = "bench/README.md"
  if not fileExists(fragPath):
    stderr.writeLine("[readme] missing " & fragPath & " -- run `nimble benchReadme` first")
    quit(1)
  spliceReadme(readmePath, machineSlug(), readFile(fragPath))

when isMainModule:
  main()
