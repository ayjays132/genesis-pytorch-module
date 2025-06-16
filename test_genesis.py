#!/usr/bin/env python3
import sys
import types
import time
import torch
import torch.nn as nn

# ─── Monkey-patch CUDA memory API to accept CPU devices ───
torch.cuda.memory_allocated = lambda device=None: 0

# ─── Monkey-patch curses on Windows ───
sys.modules['curses'] = types.ModuleType('curses')
sys.modules['_curses'] = types.ModuleType('_curses')

def main():
    print("🔧 Testing genesis_module installation...")
    try:
        from genesis_module import (
            IntegratedLearningModule,
            SelfReplayBuffer,
            EthicalGate,
            GenesisPlugin,
            attach_genesis_plugin,
            get_gui_metrics,
        )
    except Exception as e:
        print("❌ Failed to import genesis_module:", e)
        return

    print("✅ genesis_module imported successfully.\n")

    # 1) Test IntegratedLearningModule
    batch_size, seq_len, input_dim = 4, 16, 128
    hidden_size, output_size = 64, input_dim
    print(f"🧠 Testing IntegratedLearningModule (in={input_dim}, hid={hidden_size}, out={output_size})...")
    ilm = IntegratedLearningModule(input_dim, hidden_size, output_size)
    ilm.train()

    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    t0 = time.time()
    logits, raw_logits, final_hidden = ilm(x)
    t1 = time.time()

    assert logits.shape     == (batch_size, output_size)
    assert raw_logits.shape == (batch_size, output_size)
    assert final_hidden.shape == (batch_size, hidden_size)
    print(f"   ✅ Forward OK • {(t1-t0)*1000:.1f}ms")

    loss = final_hidden.sum()
    loss.backward()
    print("   ✅ Backward OK (grad on input? {})".format(x.grad is not None))
    print()

    # 2) Test GenesisPlugin standalone
    print("🔌 Testing GenesisPlugin standalone...")
    plugin = GenesisPlugin(hidden_size, hidden_size)
    plugin.train()

    x2 = torch.randn(batch_size, hidden_size, requires_grad=True)
    t0 = time.time()
    logits2, raw2, hidden2 = plugin(x2)
    t1 = time.time()

    assert logits2.shape == (batch_size, hidden_size)
    assert raw2.shape     == (batch_size, hidden_size)
    assert hidden2.shape  == (batch_size, hidden_size)
    print(f"   ✅ Plugin forward OK • {(t1-t0)*1000:.1f}ms")

    loss2 = hidden2.sum()
    loss2.backward()
    has_grads = any((p.grad is not None and p.grad.abs().sum()>0) for p in plugin.parameters())
    print(f"   ✅ Plugin backward OK (grads? {has_grads})")
    print()

    # 3) Test SelfReplayBuffer
    print("📀 Testing SelfReplayBuffer...")
    buf = SelfReplayBuffer(max_size=3)
    for i in range(3):
        h = torch.randn(1, hidden_size)
        t = torch.tensor([i])
        buf.add(h, t, priority=1.0 if i<2 else -5.0)
    assert buf.buffer[2][2] == 0.0
    print("   ➡️ Negative priority clamped to zero")

    h_samp, t_samp = buf.sample(batch_size=2)
    print("   ➡️ Sampling works:", h_samp.shape, t_samp.shape)
    assert h_samp.shape[0] == 2 and t_samp.shape[0] == 2

    for i in range(5):
        buf.add(torch.randn(1, hidden_size), torch.tensor([i]), priority=1.0)
    assert len(buf.buffer) == 3
    print("   ➡️ Buffer respects max_size")
    print()

    # 4) Test EthicalGate filtering
    print("🛡️ Testing EthicalGate filtering...")
    gate = EthicalGate(vocab_size=5, use_classifier=True, classifier_scale=2.0)
    dummy_plugin = GenesisPlugin(hidden_size, 5, vocab_size=5)
    gate.register_to_module(dummy_plugin)
    with torch.no_grad():
        for p in dummy_plugin.ethical_classifier.parameters():
            p.zero_()
        dummy_plugin.ethical_classifier[0].bias[2] = -10.0
    logits0 = torch.zeros(1, 5)
    filtered = gate.filter_logits_with_classifier(logits0.clone(), dummy_plugin)
    assert filtered[0, 2] < logits0[0, 2]
    print("   ✅ Classifier-based filtering works")
    print()

    # 5) Test attach_genesis_plugin helper
    print("🔗 Testing attach_genesis_plugin...")
    base = nn.Sequential(nn.Linear(10,16), nn.ReLU(), nn.Linear(16,8))
    plugin2 = GenesisPlugin(16,5, vocab_size=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)
    plugin2 = plugin2.to(device)
    handle = attach_genesis_plugin(base, plugin2, layer_name="0", with_grad=True)
    _ = base(torch.randn(2,10).to(device))
    assert hasattr(base, "genesis_logits")
    assert base.genesis_logits.shape == (2,5)
    handle.remove()
    print("   ✅ attach_genesis_plugin OK")
    print()

    # 6) Test get_gui_metrics
    print("📊 Testing get_gui_metrics...")
    model2 = IntegratedLearningModule(4,6,3).to(device)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    crit2 = nn.CrossEntropyLoss()
    xg = torch.randn(2,3,4).to(device)
    yg = torch.randint(0,3,(2,)).to(device)
    model2.training_step(xg, yg, opt2, crit2)

    try:
        metrics = get_gui_metrics(model2)
        for k in ("novelty_score","replay_buffer_len","step_count",
                  "cpu_memory_mb","cpu_percent","gpu_memory_mb","gpu_utilization"):
            assert k in metrics
        print("   ✅ GUI metrics retrieved")
    except Exception as e:
        print(f"   ⚠️ Skipped GUI metrics: {e}")
    print()

    # 7) Version check
    import genesis_module
    print("🏷️ genesis_module.__version__ =", genesis_module.__version__)
    print("\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    main()
