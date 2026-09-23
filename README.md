# PartiGen

Source snapshot of the two components used for text-conditioned G1 motion generation and GRIT tracking/deployment.

| Component | Entry point |
| --- | --- |
| Motion generation: BP-MVAE, duration-adaptive latent diffusion, frame-level EOS | [TextOpRobotMDAR](TextOpRobotMDAR/README.md) |
| GRIT tracking, simulation and robot deployment | [GRIT_teleop_deploy](GRIT_teleop_deploy/README.md) · [中文说明](GRIT_teleop_deploy/README_ZH.md) |

## Training configuration

The PartiGen training entry points are `train_partigen_mvae` and `train_partigen_dar`. Experiment hyperparameters remain mandatory (`???`); researchers must provide them explicitly. Legacy experiment configurations are retained as historical code and are not the new training defaults.

```bash
cd TextOpRobotMDAR
python -m pip install -e .
python -m robotmdar.cli --config-name train_partigen_mvae --cfg job
python -m robotmdar.cli --config-name train_partigen_dar --cfg job
```

The configuration commands only display templates. Training also requires the external dataset, robot assets, CLIP/Isaac dependencies and a complete experiment configuration described in the component README.

## GRIT setup

Follow the GRIT component documentation for its separate Python environment and C++ bridge. This snapshot retains runtime source, robot XML/meshes, vendored libraries and their notices. Policy weights and motion arrays must be supplied separately.

The `qpos_to_grit_npz.py` conversion tool additionally expects `description/robots/g1/g1_23dof_lock_wrist.xml` beside `GRIT_teleop_deploy/`. That external asset directory is not one of the two source directories packaged here. Some original local manuals also refer to workstation-specific parent scripts and generated result folders; those artifacts are not included.

## Snapshot contents

- Includes current working-tree source, including GRIT changes that had not been committed in the original checkout.
- Excludes original Git history, virtual environments, caches, build output, datasets, policy/checkpoint weights, generated motion arrays, evaluation output and recordings.
- Preserves small runtime assets, upstream documentation media, dependency lockfiles, copyright comments and third-party notices.
- The local manuscript PDF is not included in this source snapshot.

See [PACKAGING.md](PACKAGING.md) and [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) for the packaging boundary and checksums.

This repository snapshot does not claim completed paper metric reproduction or a validated hardware deployment. TextOpRobotMDAR's source tests passed locally; real G1 assets and server checkpoints were not validated as part of packaging.

## Attribution

Existing component license declarations and copyright notices are preserved. The motion component declares MIT in `pyproject.toml`, but its provided directory does not include a standalone license file. GRIT third-party terms are documented in [THIRD_PARTY.md](GRIT_teleop_deploy/THIRD_PARTY.md) and the vendored license files. This combined snapshot does not introduce a new blanket license for third-party code or assets.
