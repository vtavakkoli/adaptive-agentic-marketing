# Migration note: `adaptive_full` → `adaptive_framework`

- Legacy naming has been refactored.
- `adaptive_framework` now refers to the previous flat `adaptive_full` framework.
- `adaptive_hierarchical` is the new hierarchical, cost-sensitive, uncertainty-aware framework.
- Internal aliasing keeps `adaptive_full` runnable for backward compatibility, but user-facing reports no longer present it as an active mode.
