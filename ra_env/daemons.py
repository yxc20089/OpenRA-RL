"""Docker daemons behind the addresses: find the container and restart it.

The daemon's memory grows with every game it plays (~2-8 MB, more with frequent
decisions), so long runs restart each daemon every so many games.
"""

import subprocess
import time

import grpc


def container_for(address: str) -> str:
    """Name of the running container publishing this address's port ("" if none)."""
    port = address.rsplit(":", 1)[-1]
    out = subprocess.run(["docker", "ps", "--filter", f"publish={port}", "--format", "{{.Names}}"],
                         capture_output=True, text=True, timeout=30, check=False).stdout.split()
    return out[0] if out else ""


def wait_reachable(address: str, timeout_s: float = 120.0) -> bool:
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        return True
    except grpc.FutureTimeoutError:
        return False
    finally:
        channel.close()


def restart(address: str, timeout_s: float = 120.0) -> bool:
    """Restart the daemon behind `address` and wait until it accepts connections."""
    name = container_for(address)
    if not name:
        return False
    subprocess.run(["docker", "restart", name], capture_output=True, timeout=120, check=False)
    ok = wait_reachable(address, timeout_s)
    time.sleep(3)  # gRPC is up before the mod finishes loading
    return ok
