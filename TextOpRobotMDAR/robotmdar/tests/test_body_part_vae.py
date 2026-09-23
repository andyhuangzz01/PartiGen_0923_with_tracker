"""
Test script for Body Part Attention VAE

This script verifies that the BodyPartAutoMldVae implementation:
1. Can be instantiated correctly
2. Forward/backward passes work
3. Is compatible with existing training pipeline
4. Body part attention is properly configured

Usage:
    python -m robotmdar.tests.test_body_part_vae
    
    Or from project root:
    python TextOpRobotMDAR/robotmdar/tests/test_body_part_vae.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_body_part_attention():
    """Test BodyPartMultiheadAttention module"""
    print("=" * 60)
    print("Testing BodyPartMultiheadAttention...")
    print("=" * 60)
    
    from robotmdar.model.operator.body_part_attention import BodyPartMultiheadAttention
    
    # Parameters
    d_model = 512
    nhead = 4
    num_body_parts = 4
    seq_len = 20
    batch_size = 8
    
    # Create module
    attn = BodyPartMultiheadAttention(
        d_model=d_model,
        nhead=nhead,
        num_body_parts=num_body_parts,
        dropout=0.1,
    )
    
    # Test forward pass
    x = torch.randn(seq_len, batch_size, d_model)
    output, _ = attn(x, x, x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Forward pass: input {x.shape} -> output {output.shape}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")
    
    # Check body part embeddings
    print(f"✓ Body part embeddings shape: {attn.body_part_embeddings.shape}")
    print(f"  - num_body_parts: {num_body_parts}")
    print(f"  - heads_per_part: {attn.heads_per_part}")
    
    print("\n✓ BodyPartMultiheadAttention test passed!\n")
    return True


def test_body_part_encoder_layer():
    """Test BodyPartTransformerEncoderLayer"""
    print("=" * 60)
    print("Testing BodyPartTransformerEncoderLayer...")
    print("=" * 60)
    
    from robotmdar.model.operator.body_part_attention import BodyPartTransformerEncoderLayer
    
    d_model = 512
    nhead = 4
    seq_len = 20
    batch_size = 8
    
    layer = BodyPartTransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="gelu",
        num_body_parts=4,
    )
    
    x = torch.randn(seq_len, batch_size, d_model)
    output = layer(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Forward pass: input {x.shape} -> output {output.shape}")
    
    # Test with position encoding
    pos = torch.randn(seq_len, batch_size, d_model)
    output_with_pos = layer(x, pos=pos)
    assert output_with_pos.shape == x.shape
    print("✓ Forward pass with position encoding")
    
    print("\n✓ BodyPartTransformerEncoderLayer test passed!\n")
    return True


def test_body_part_skip_encoder():
    """Test BodyPartSkipTransformerEncoder"""
    print("=" * 60)
    print("Testing BodyPartSkipTransformerEncoder...")
    print("=" * 60)
    
    from robotmdar.model.operator.body_part_attention import (
        BodyPartTransformerEncoderLayer,
        BodyPartSkipTransformerEncoder,
    )
    
    d_model = 512
    nhead = 4
    num_layers = 9  # Must be odd
    seq_len = 20
    batch_size = 8
    
    encoder_layer = BodyPartTransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=1024,
        dropout=0.1,
        num_body_parts=4,
    )
    
    encoder = BodyPartSkipTransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        norm=nn.LayerNorm(d_model),
    )
    
    x = torch.randn(seq_len, batch_size, d_model)
    output = encoder(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Forward pass: input {x.shape} -> output {output.shape}")
    print(f"  - num_layers: {num_layers}")
    print(f"  - input_blocks: {len(encoder.input_blocks)}")
    print(f"  - output_blocks: {len(encoder.output_blocks)}")
    
    # Test backward
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\n✓ BodyPartSkipTransformerEncoder test passed!\n")
    return True


def test_body_part_vae():
    """Test full BodyPartAutoMldVae"""
    print("=" * 60)
    print("Testing BodyPartAutoMldVae...")
    print("=" * 60)
    
    from robotmdar.model.body_part_mld_vae import BodyPartAutoMldVae
    
    # Parameters matching real config
    nfeats = 57  # 23-DOF robot
    latent_dim = [1, 128]
    h_dim = 512
    ff_size = 1024
    num_layers = 9
    num_heads = 4
    batch_size = 4
    history_len = 8
    future_len = 8
    
    # Create model
    vae = BodyPartAutoMldVae(
        nfeats=nfeats,
        latent_dim=latent_dim,
        h_dim=h_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        arch="all_encoder",
        num_body_parts=4,
        use_body_part_attention=True,
    )
    
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Test encode
    history_motion = torch.randn(batch_size, history_len, nfeats)
    future_motion = torch.randn(batch_size, future_len, nfeats)
    
    latent, dist = vae.encode(future_motion, history_motion)
    print(f"✓ Encode: history {history_motion.shape}, future {future_motion.shape} -> latent {latent.shape}")
    
    # Test decode
    decoded = vae.decode(latent, history_motion, nfuture=future_len)
    assert decoded.shape == future_motion.shape, f"Decoded shape mismatch: {decoded.shape} vs {future_motion.shape}"
    print(f"✓ Decode: latent {latent.shape} -> decoded {decoded.shape}")
    
    # Test backward
    loss = (decoded - future_motion).pow(2).mean()
    loss.backward()
    print(f"✓ Backward pass successful, loss: {loss.item():.6f}")
    
    # Print body part info
    info = vae.get_body_part_info()
    print(f"\nBody Part Configuration:")
    for key, value in info.items():
        print(f"  - {key}: {value}")
    
    print("\n✓ BodyPartAutoMldVae test passed!\n")
    return True


def test_compatibility_with_standard_vae():
    """Test that BodyPartAutoMldVae has same interface as AutoMldVae"""
    print("=" * 60)
    print("Testing API compatibility with AutoMldVae...")
    print("=" * 60)
    
    from robotmdar.model.mld_vae import AutoMldVae
    from robotmdar.model.body_part_mld_vae import BodyPartAutoMldVae
    
    # Same parameters
    params = dict(
        nfeats=57,
        latent_dim=[1, 128],
        h_dim=512,
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        arch="all_encoder",
    )
    
    standard_vae = AutoMldVae(**params)
    bodypart_vae = BodyPartAutoMldVae(**params, use_body_part_attention=True)
    
    # Test same interface
    batch_size = 2
    history_len = 8
    future_len = 8
    nfeats = 57
    
    history = torch.randn(batch_size, history_len, nfeats)
    future = torch.randn(batch_size, future_len, nfeats)
    
    # Both should have same encode/decode signature
    std_latent, std_dist = standard_vae.encode(future, history)
    bp_latent, bp_dist = bodypart_vae.encode(future, history)
    
    assert std_latent.shape == bp_latent.shape, "Latent shapes don't match!"
    print(f"✓ Encode interface compatible: latent shape {std_latent.shape}")
    
    std_decoded = standard_vae.decode(std_latent, history, future_len)
    bp_decoded = bodypart_vae.decode(bp_latent, history, future_len)
    
    assert std_decoded.shape == bp_decoded.shape, "Decoded shapes don't match!"
    print(f"✓ Decode interface compatible: output shape {std_decoded.shape}")
    
    print("\n✓ API compatibility test passed!\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("BODY PART ATTENTION VAE TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        ("Body Part Attention", test_body_part_attention),
        ("Body Part Encoder Layer", test_body_part_encoder_layer),
        ("Body Part Skip Encoder", test_body_part_skip_encoder),
        ("Body Part VAE", test_body_part_vae),
        ("API Compatibility", test_compatibility_with_standard_vae),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n✗ {name} test FAILED with error: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
