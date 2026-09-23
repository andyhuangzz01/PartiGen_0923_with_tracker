"""
Quick check of train.pkl data structure
"""
import joblib
from pathlib import Path

train_pkl = Path('./dataset/HumanML3D-G1-23DOF-30fps/train.pkl')
print(f"Loading {train_pkl}...")

data = joblib.load(train_pkl)
print(f"Loaded {len(data)} sequences")
print(f"Data type: {type(data)}")

# Check first sequence
first_seq = data[0]
print(f"\nFirst sequence type: {type(first_seq)}")
print(f"First sequence keys: {first_seq.keys() if isinstance(first_seq, dict) else 'Not a dict'}")

if isinstance(first_seq, dict):
    for key, value in first_seq.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype if hasattr(value, 'dtype') else type(value)}")
        elif hasattr(value, '__len__'):
            print(f"  {key}: len={len(value)}, type={type(value)}")
        else:
            print(f"  {key}: {type(value)}, value={value if not isinstance(value, (list, dict)) else '...'}")
    
    # Check motion field
    if 'motion' in first_seq:
        motion = first_seq['motion']
        print(f"\nMotion field:")
        print(f"  Type: {type(motion)}")
        if isinstance(motion, dict):
            print(f"  Keys: {motion.keys()}")
            for k, v in motion.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}")
                else:
                    print(f"    {k}: type={type(v)}")
        elif hasattr(motion, 'shape'):
            print(f"  Shape: {motion.shape}")
            print(f"  Dtype: {motion.dtype if hasattr(motion, 'dtype') else type(motion)}")
