"""FastAPI application for the OpenRA-RL environment.

Creates the OpenEnv-compatible server using create_app().
"""

from openenv.core.env_server import create_app

from openra_env.models import OpenRAAction, OpenRAObservation
from openra_env.server.openra_environment import OpenRAEnvironment

app = create_app(
    OpenRAEnvironment,
    OpenRAAction,
    OpenRAObservation,
    env_name="openra_env",
)


def main():
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )


if __name__ == "__main__":
    main()
