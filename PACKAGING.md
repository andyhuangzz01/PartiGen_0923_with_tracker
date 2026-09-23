# Packaging record

This combined repository contains a fresh snapshot of the current files, without importing either component's original Git history. Original working directories were not modified by packaging. Source file SHA-256 checksums and executable/symlink metadata are recorded in `SOURCE_MANIFEST.json`; root packaging documents are additional files.

| Component | Source files |
| --- | ---: |
| TextOpRobotMDAR | 131 |
| GRIT_teleop_deploy | 1379 |
| Total | 1510 |

The source snapshot is 133,853,227 bytes before Git/ZIP compression. Four relative Linux symlinks are preserved as Git symlinks and ZIP Unix symlink entries. Vendored Unitree/DDS runtime libraries, robot XML/STL assets, and original copyright/license notices remain included. The snapshot includes 35 GRIT tools and current uncommitted runtime/test/script changes.

Excluded: virtual environments, dependency caches, `.git` directories, builds, generated videos/reports/results, motion arrays (`.pkl`, `.npy`, `.npz`), model weights (`.pt`, `.pth`, `.ckpt`, `.onnx`), local environment/credential files, and the local manuscript PDF. Original documentation GIFs and checkpoint metadata/README are retained.

No training hyperparameter defaults were added. New MVAE/DAR experiment templates still require explicit configuration. Legacy component files retain their existing values and paths.

External runtime requirements remain documented in the component READMEs. Supply GRIT policy weights and motion files separately. The conversion tool also expects the external `description/robots/g1/g1_23dof_lock_wrist.xml` tree at the repository root. Original local manuals may refer to parent-level workstation scripts, data directories or reports which are not part of this two-directory source snapshot.

Checks: source-to-snapshot hashes, absence of included datasets/checkpoints/environments, credential-pattern scan, executable and symlink metadata, ZIP integrity, and remote commit/privacy verification. Packaging does not execute a robot, start training or validate server checkpoints.
