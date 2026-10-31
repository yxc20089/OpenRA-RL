# OpenRA RL daemon: the forked engine (OpenRA/ submodule) running headless in
# multi-session mode. The container is C# only; Python talks gRPC on :9999.
#
#   git submodule update --init OpenRA
#   docker compose up -d --build

# ------------------------------------------------------------------------------
# Build the engine from the local submodule
# ------------------------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    make git ca-certificates \
    libsdl2-dev libopenal-dev libfreetype-dev liblua5.1-0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY OpenRA/ /src/openra/
WORKDIR /src/openra

# A Windows checkout leaves CRLF in Makefile/.cs/.yaml, which breaks make and MiniYaml.
RUN find . -type f \( \
        -name '*.sh' -o -name 'Makefile' -o -name '*.mk' \
        -o -name '*.cs' -o -name '*.csproj' -o -name '*.props' -o -name '*.targets' \
        -o -name '*.sln' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \
        -o -name '*.lua' -o -name '*.proto' -o -name '*.frag' -o -name '*.vert' \) \
        -exec sed -i 's/\r$//' {} + && \
    find . -name '*.sh' -exec chmod +x {} +

# SKIP_PROTOC stays unset: Grpc.Tools regenerates the C# stubs from the proto.
RUN make TARGETPLATFORM=unix-generic CONFIGURATION=Release && \
    test -f bin/OpenRA.dll && test -f bin/OpenRA.Platforms.Null.dll

# ------------------------------------------------------------------------------
# Runtime: ASP.NET Core (the gRPC server is Kestrel) + native libs OpenRA loads
# ------------------------------------------------------------------------------
FROM mcr.microsoft.com/dotnet/aspnet:8.0-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 libopenal1 libfreetype6 liblua5.1-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/openra/bin /opt/openra/bin
COPY --from=build /src/openra/mods /opt/openra/mods
COPY --from=build /src/openra/glsl /opt/openra/glsl
COPY --from=build ["/src/openra/global mix database.dat", "/opt/openra/global mix database.dat"]

RUN LIBDIR=/usr/lib/$(uname -m)-linux-gnu && \
    ln -sf "$LIBDIR/libSDL2-2.0.so.0" /opt/openra/bin/SDL2.so && \
    ln -sf "$LIBDIR/libopenal.so.1" /opt/openra/bin/soft_oal.so && \
    ln -sf "$LIBDIR/libfreetype.so.6" /opt/openra/bin/freetype6.so && \
    ln -sf "$LIBDIR/liblua5.1.so.0" /opt/openra/bin/lua51.so

# OPENRA_RL_DETERMINISTIC: seed bot LocalRandom from the game seed (reproducible episodes).
# OPENRA_RL_AUTOSTART=0: never spawn the v2 in-game Python sidecar.
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_ROLL_FORWARD=LatestMajor \
    OPENRA_RL_DETERMINISTIC=1 \
    OPENRA_RL_AUTOSTART=0 \
    OPENRA_RL_FAST_ADVANCE_DEADLINE_S=90

WORKDIR /opt/openra
EXPOSE 9999

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD bash -c '</dev/tcp/127.0.0.1/9999' || exit 1

ENTRYPOINT ["dotnet", "/opt/openra/bin/OpenRA.dll", "Engine.EngineDir=/opt/openra", "Game.Mod=ra", \
            "Game.Platform=Null", "Launch.MultiSession=9999"]
