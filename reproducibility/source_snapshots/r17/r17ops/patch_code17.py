"""R17 code = R16 code (src16) + F2 mode 'regime' (frozen 2026-10-17): the HIGH-decode step budget applies only in the
short regime (protected parameters) and is off in the long regime. Usage: patch_code17.py SRC16 DST R17DIR"""
import shutil, sys
from pathlib import Path
SRC, DST, R17 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(R17 / 'code_add' / 'test_r17.py', DST / 'code' / 'test_r17.py')


def edit(path, pairs):
    p = DST / path; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    p.write_text(s)


edit('code/r16_policy.py', [
    ("MODES, F2 = ('protected', 'hyst', 'adapt'), ('on', 'off', 'pressure')",
     "MODES, F2 = ('protected', 'hyst', 'adapt'), ('on', 'off', 'pressure', 'regime')"),
    ("  pressure   cap only while held KV >= theta_low * K (memory-pressure regime)\n",
     "  pressure   cap only while held KV >= theta_low * K (memory-pressure regime)\n"
     "  regime     (R17) cap only in the short regime (protected parameters); no cap in the long regime\n"),
])
edit('code/runtime_r16.py', [
    ("""    def _gr_token_budget(self, budget):
""", """    def _gr_token_budget(self, budget):
        if self._gr.f2 == 'regime' and self._gr.long:                 # R17: F2 follows the regime
            cap = self._gr_budget_cap
            if budget > cap and any(r.priority == 0 and r.num_output_tokens > 0 for r in self.running):
                self._r16_cap_skipped += 1
            return budget
"""),
])
print('patched17', DST)
