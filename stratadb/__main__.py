"""Command line: ``stratadb serve | stats | compact | checkpoint``."""

from __future__ import annotations

import argparse
import json
import logging
import os

from . import __version__
from .database import Database


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="stratadb", description="StrataDB graph-native vector database")
    parser.add_argument("--version", action="version", version=f"stratadb {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the HTTP server")
    p.add_argument("--path", default="./data", help="database directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7700)
    p.add_argument("--api-key", default=os.environ.get("STRATADB_API_KEY"),
                   help="require 'Authorization: Bearer <key>' (or set STRATADB_API_KEY)")
    p.add_argument("-v", "--verbose", action="store_true", help="log every request")

    for name, help_text in (("stats", "print collection stats"),
                            ("compact", "compact segments"),
                            ("merge", "merge all sealed segments into one"),
                            ("checkpoint", "write a checkpoint and empty the WAL")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--path", default="./data")
        p.add_argument("collection", nargs="?", help="collection name (default: all)")

    p = sub.add_parser("backup", help="write a consistent copy of the database")
    p.add_argument("--path", default="./data")
    p.add_argument("--out", required=True, help="destination directory (must not exist)")

    p = sub.add_parser("export", help="write a collection as JSON lines")
    p.add_argument("--path", default="./data")
    p.add_argument("collection")
    p.add_argument("--out", required=True, help="output .jsonl file")

    p = sub.add_parser("import", help="load JSON lines into a collection (created if missing)")
    p.add_argument("--path", default="./data")
    p.add_argument("collection")
    p.add_argument("--in", dest="infile", required=True, help="input .jsonl file")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                            format="%(asctime)s %(levelname)s %(message)s")
        from .server import serve
        serve(args.path, args.host, args.port, args.api_key)
        return

    if args.cmd == "export":
        with Database(args.path) as db, open(args.out, "w", encoding="utf-8") as fp:
            counts = db[args.collection].export_jsonl(fp)
        print(json.dumps({"exported": counts, "to": args.out}, indent=2))
        return

    if args.cmd == "import":
        with open(args.infile, encoding="utf-8") as fp:
            header = json.loads(fp.readline())
            if header.get("type") != "collection":
                raise SystemExit("import file must start with a collection header line")
            fp.seek(0)
            with Database(args.path) as db:
                col = db.create_collection(args.collection, dim=int(header["dim"]), metric=header["metric"])
                counts = col.import_jsonl(fp)
        print(json.dumps({"imported": counts, "into": args.collection}, indent=2))
        return

    if args.cmd == "backup":
        with Database(args.path) as db:
            names = db.backup(args.out)
        print(json.dumps({"backed_up": names, "to": args.out}, indent=2))
        return

    with Database(args.path) as db:
        names = [args.collection] if args.collection else db.list_collections()
        out = {}
        for name in names:
            col = db[name]
            if args.cmd == "compact":
                out[name] = col.compact()
            elif args.cmd == "merge":
                out[name] = col.merge(1)
            elif args.cmd == "checkpoint":
                out[name] = {"lsn": col.checkpoint()}
            else:
                out[name] = col.stats()
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
