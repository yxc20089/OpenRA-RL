"""Regenerate ra_env/_proto from proto/rl_bridge.proto.

Run after editing the proto (the C# side regenerates its own stubs at build):
    python scripts/gen_proto.py
"""

import pathlib
import re
import sys

from grpc_tools import protoc

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "ra_env" / "_proto"


def main() -> int:
    rc = protoc.main([
        "protoc",
        f"-I{ROOT / 'proto'}",
        f"--python_out={OUT}",
        f"--grpc_python_out={OUT}",
        str(ROOT / "proto" / "rl_bridge.proto"),
    ])
    if rc != 0:
        return rc
    # protoc emits a top-level import; make it package-relative.
    grpc_file = OUT / "rl_bridge_pb2_grpc.py"
    src = grpc_file.read_text(encoding="utf-8")
    src = re.sub(r"^import rl_bridge_pb2 as", "from . import rl_bridge_pb2 as", src, flags=re.MULTILINE)
    grpc_file.write_text(src, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
