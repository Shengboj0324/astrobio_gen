#!/usr/bin/env python3
"""
Test PyTorch Geometric functionality
"""

import sys
import os
sys.path.append(os.getcwd())

print("Testing PyTorch Geometric...")

try:
    import torch_geometric
    print(f"✅ PyTorch Geometric imported: version {torch_geometric.__version__}")
    
    # Test basic imports
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    print("✅ Basic PyTorch Geometric components imported")
    
    # Test creating a simple graph
    import torch
    edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    print(f"✅ Graph data created: {data}")
    
    # Test GCN layer
    conv = GCNConv(1, 16)
    out = conv(x, edge_index)
    print(f"✅ GCN forward pass successful: {out.shape}")
    
    # Test our training orchestrator
    try:
        from training.enhanced_training_orchestrator import EnhancedTrainingOrchestrator
        print("✅ Training orchestrator import successful")
    except Exception as e:
        print(f"❌ Training orchestrator import failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ PyTorch Geometric test failed: {e}")
    import traceback
    traceback.print_exc()
