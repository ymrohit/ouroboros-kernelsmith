# Best Kernels

This directory contains the 69 kernels from the final V2 stability gate.

Source of truth:

- Kernel list: `reports/rebench_stability_v2.json`
- Summary report: `reports/rebench_stability_v2.md`
- Original source before curation: the local Modal volume snapshot, which is intentionally
  not tracked in this public repo.

Only the 69 ops present in `reports/rebench_stability_v2.json["per_op"]` are included here.
The broader local volume backup also contains probe and follow-up kernels that are useful for
private analysis but are not part of the final 69-kernel stability claim.
