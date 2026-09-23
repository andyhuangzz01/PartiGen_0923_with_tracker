"""
Debug script to test Body Part VAE configuration loading

This script helps diagnose configuration and import issues.
"""

import sys
from pathlib import Path

print("=" * 60)
print("Body Part VAE Configuration Test")
print("=" * 60)

# Test 1: Import body_part_mld_vae
print("\n1. Testing import of body_part_mld_vae...")
try:
    from robotmdar.model.body_part_mld_vae import BodyPartAutoMldVae
    print("   ✓ BodyPartAutoMldVae imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import body_part_attention modules
print("\n2. Testing import of body_part_attention...")
try:
    from robotmdar.model.operator.body_part_attention import (
        BodyPartMultiheadAttention,
        BodyPartTransformerEncoderLayer,
        BodyPartSkipTransformerEncoder,
    )
    print("   ✓ Body part attention modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check configuration file
print("\n3. Checking configuration files...")
config_dir = Path("robotmdar/config")
vae_config = config_dir / "vae" / "body_part_skip_vae.yaml"
train_config = config_dir / "train_mvae_humanml3d_23dof_bodypart.yaml"

if vae_config.exists():
    print(f"   ✓ VAE config exists: {vae_config}")
else:
    print(f"   ✗ VAE config missing: {vae_config}")

if train_config.exists():
    print(f"   ✓ Train config exists: {train_config}")
else:
    print(f"   ✗ Train config missing: {train_config}")

# Test 4: Try loading config with Hydra
print("\n4. Testing Hydra configuration loading...")
try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    config_path = str(config_dir.absolute())
    print(f"   Config path: {config_path}")
    
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name="train_mvae_humanml3d_23dof_bodypart")
        print("   ✓ Configuration loaded successfully")
        print(f"   - expname: {cfg.expname}")
        print(f"   - task: {cfg.task}")
        print(f"   - nfeats: {cfg.nfeats}")
        print(f"   - vae._target_: {cfg.vae._target_}")
        print(f"   - use_body_part_attention: {cfg.vae.use_body_part_attention}")
        print(f"   - num_body_parts: {cfg.vae.num_body_parts}")
except Exception as e:
    print(f"   ✗ Configuration loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Try instantiating the model
print("\n5. Testing model instantiation...")
try:
    import torch
    
    model = BodyPartAutoMldVae(
        nfeats=57,
        latent_dim=[1, 128],
        h_dim=512,
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        arch="all_encoder",
        use_body_part_attention=True,
        num_body_parts=4,
    )
    print("   ✓ Model instantiated successfully")
    
    # Test forward pass
    history = torch.randn(2, 8, 57)
    future = torch.randn(2, 8, 57)
    latent, dist = model.encode(future, history)
    print(f"   ✓ Encode test passed: latent shape {latent.shape}")
    
    decoded = model.decode(latent, history, nfuture=8)
    print(f"   ✓ Decode test passed: output shape {decoded.shape}")
    
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! Body Part VAE is ready to use.")
print("=" * 60)
print("\nYou can now run:")
print("  python -m robotmdar.train.train_mvae --config-name train_mvae_humanml3d_23dof_bodypart")
