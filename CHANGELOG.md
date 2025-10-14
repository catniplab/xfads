# Change Log

- 2025-10-14
  - Initialized change log and documented logging workflow in `AGENTS.md`.
  - Removed `memory_cleanup` decorator usage and supporting definitions from the package.
  - Updated smoother modules and prebuilt models to use correct relative imports and device-aware tensor creation.
  - Renamed agent logging section in `AGENTS.md` to clarify change log policy.
  - Registered fixed dynamics noise as buffers and ensured smoother masks, padding, and randomness build on the active device.
  - Removed duplicate `lr_gamma_decay` override from the LDS Lightning example config.
  - Dropped explicit device arguments from utility dynamics modules and harmonized GRU dynamics construction across examples.
  - Switched LDS Lightning example to auto-select accelerators/devices and refreshed training defaults.
  - Set LDS example dataloaders to use configurable workers/pin-memory defaults compatible with sandboxed CPU runs.
