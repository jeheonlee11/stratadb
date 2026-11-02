"""Talk to a running StrataDB server.

Start one first:

    stratadb serve --path ./data --port 7700 --api-key dev-key

then:

    python examples/client_server.py
"""

import numpy as np

from stratadb.client import Client

client = Client("http://127.0.0.1:7700", api_key="dev-key")
print(client.health())

notes = client.create_collection("notes", dim=8)
rng = np.random.default_rng(1)
vecs = rng.normal(size=(5, 8))
notes.upsert(["n1", "n2", "n3", "n4", "n5"], vecs, [{"tag": t} for t in "aabba"])
notes.link([{"src": "n1", "dst": "n4", "rel": "see_also"}])

print(notes.search(vecs[0], k=2, filter={"tag": "a"}, expand_hops=1))
print(notes.stats())
