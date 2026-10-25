from pathlib import Path
import hashlib,json,urllib.request
root=Path(__file__).resolve().parent
receipt=json.loads((root/'receipt.json').read_text())
dest=root/'vendor_reference/pfs_ae_d93ff69';dest.mkdir(parents=True,exist_ok=True)
for name,r in receipt['official_sources'].items():
    raw=urllib.request.urlopen(r['url'],timeout=30).read()
    assert hashlib.sha256(raw).hexdigest()==r['sha256'],name
    (dest/name).write_bytes(raw)
print('Fetched exact checksum-pinned public AE class sources.')
