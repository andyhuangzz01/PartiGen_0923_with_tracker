"""
Check the raw contact_mask values in train.pkl
"""
import joblib
import numpy as np
from pathlib import Path

train_pkl = Path('./dataset/HumanML3D-G1-23DOF-30fps/train.pkl')
print(f"Loading {train_pkl}...")

data = joblib.load(train_pkl)
print(f"Loaded {len(data)} sequences\n")

# Check contact values in first 10 sequences
print("="*80)
print("CHECKING CONTACT_MASK VALUES IN RAW DATA")
print("="*80)

all_contacts = []
for i, seq in enumerate(data[:20]):
    motion_dict = seq['motion']
    contact = motion_dict['contact_mask']
    
    if not isinstance(contact, np.ndarray):
        contact = np.array(contact)
    
    all_contacts.append(contact)
    
    if i < 5:
        print(f"\nSequence {i}:")
        print(f"  Shape: {contact.shape}")
        print(f"  Dtype: {contact.dtype}")
        print(f"  Min: {contact.min():.6f}, Max: {contact.max():.6f}")
        print(f"  Mean: {contact.mean():.6f}, Std: {contact.std():.6f}")
        print(f"  Unique values (first 10): {np.unique(contact)[:10]}")
        print(f"  First 5 frames:\n{contact[:5]}")

# Overall statistics
all_contacts_concat = np.concatenate(all_contacts, axis=0)
print("\n" + "="*80)
print("OVERALL STATISTICS (first 20 sequences)")
print("="*80)
print(f"Total frames: {len(all_contacts_concat)}")
print(f"Min: {all_contacts_concat.min():.6f}")
print(f"Max: {all_contacts_concat.max():.6f}")
print(f"Mean: {all_contacts_concat.mean(axis=0)}")
print(f"Std: {all_contacts_concat.std(axis=0)}")
print(f"\nValue distribution:")
print(f"  Unique values: {len(np.unique(all_contacts_concat))}")
if len(np.unique(all_contacts_concat)) < 20:
    print(f"  Values: {np.unique(all_contacts_concat)}")
else:
    print(f"  Min 10: {np.unique(all_contacts_concat)[:10]}")
    print(f"  Max 10: {np.unique(all_contacts_concat)[-10:]}")

# Check if binary
is_binary = np.all(np.isin(all_contacts_concat, [0, 1]))
print(f"\nIs binary (only 0 and 1)? {is_binary}")

if not is_binary:
    print("\n⚠ WARNING: contact_mask is NOT binary!")
    print("  This explains why the mean is ~0.994 instead of ~0.5")
    print("  The data might be incorrectly formatted or normalized.")
