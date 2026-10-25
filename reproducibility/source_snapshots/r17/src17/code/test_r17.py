"""R17 offline tests: F2 'regime' mode (F2 on in the short regime, off in the long regime)."""
import importlib, sys, types, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_r16 as T
from r16_policy import R16Controller


class RegimeF2Tests(unittest.TestCase):
    def setUp(self):
        stub = types.ModuleType('runtime_r15')

        class R15Scheduler:
            def _gr_token_budget(self, budget): return 'parent'
            def _gr_emit(self, event, **f): pass
        stub.R15Scheduler = R15Scheduler
        self.saved = sys.modules.get('runtime_r15'); sys.modules['runtime_r15'] = stub
        sys.modules.pop('runtime_r16', None); self.mod = importlib.import_module('runtime_r16')

    def tearDown(self):
        sys.modules.pop('runtime_r16', None)
        if self.saved is not None: sys.modules['runtime_r15'] = self.saved
        else: sys.modules.pop('runtime_r15', None)

    def make(self, mode, long):
        s = self.mod.R16Scheduler.__new__(self.mod.R16Scheduler)
        s._gr = R16Controller(T.r16(mode, 'regime'), 16, 400, T.Clock()); s._gr.long = long
        s._gr.held_fn = lambda: 390; s._gr_budget_cap = 512; s._r16_cap_skipped = 0
        s.running = [types.SimpleNamespace(priority=0, num_output_tokens=3)]
        return s

    def test_regime_gate(self):
        self.assertEqual(self.make('adapt', False)._gr_token_budget(4096), 'parent')     # short: F2 on
        lo = self.make('adapt', True)
        self.assertEqual(lo._gr_token_budget(4096), 4096)                                # long: F2 off
        self.assertEqual(lo._r16_cap_skipped, 1)
        self.assertEqual(self.make('protected', False)._gr_token_budget(4096), 'parent')
        self.assertEqual(self.make('hyst', True)._gr_token_budget(4096), 4096)

    def test_controller_accepts_regime(self):
        c = R16Controller(T.r16('adapt', 'regime'), 16, 400, T.Clock())
        self.assertEqual(c.f2, 'regime')


if __name__ == '__main__':
    unittest.main()
