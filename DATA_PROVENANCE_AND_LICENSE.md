# Data provenance, assumptions and redistribution gate

> **DO NOT MERGE. Do not publish third-party materials before the license and redistribution rights have been reviewed.**

| Material | Origin / nature | Status before public upload |
|---|---|---|
| 2035 PWF bus/branch tables | EPE planning/network export, transformed for this study | **Requires redistribution review.** Check EPE license/terms and original attribution. |
| Station-to-bus crosswalk | ONS station positions and EPE bus mapping | **Requires redistribution review.** |
| MG boundary coordinates | Derived from geographic administrative data | Verify source and geographic-data license. |
| Historical dispatch, generation, hydrology/wind/load | ONS/EPE and prior preprocessing | Verify attribution, download terms and provenance. |
| Resource candidate areas and allocation weights | **Model assumptions** | Do not identify as independently measured resource reserves. |
| Equipment and routing-cost coefficients | Mix of cited reference values and assumed engineering coefficients | Attribute source fields and distinguish from assumptions. |
| Scenario optimization CSV/JSON results | Derived computational outputs | Preserve input provenance and associated limitations. |
| Python optimization scripts | Research scripts from working study | Confirm authorship and applicable code license before declaring a license. |

**Release gate:** verify every input file, original source URL, access date, source license, and rights to publish modified/derived data. Where permission is unavailable, commit download instructions and schema rather than the protected raw files. The review ZIP is not an authorization to redistribute.

**Model boundaries:** 12 representative candidate regions, 4-area northern case study, 2035 1,489-bus/1,890-branch PWF; 2050 frozen-topology stress scenario. Only 283 filtered physical high-voltage lines are monitored by the main DC thermal screen.

## What was separated in this handoff

The six input files in `PRIVATE_INPUTS_FOR_LOCAL_REPRODUCTION.zip` are not in the `01_GITHUB_REVIEW_BRANCH_UPLOAD` folder. Their source/redistribution status is unresolved and they are for local, authorized reproducibility only. **Do not force-add ignored inputs or original source archives.** Remaining derived result tables also require review before a public push.
