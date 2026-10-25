import random
from pfs_reference import completion_peak
rng=random.Random(20261015)
for case in range(1000):
    pairs=[(rng.randint(1,4096),rng.randint(0,80)) for _ in range(rng.randint(1,16))]
    B=rng.choice([1,4,8,16,32])
    expected=max(sum((base+t+B-1)//B for base,left in pairs if left>=t) for t in range(max(x[1] for x in pairs)+1))
    assert completion_peak(pairs,B)==expected,case
print('PASS: 1000 exhaustive block-peak cases; no GPU required.')
