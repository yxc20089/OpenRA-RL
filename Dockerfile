# ==============================================================================
# Stage 1: Build OpenRA from source (C#/.NET 8.0)
# ==============================================================================
FROM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim AS openra-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    git \
    libsdl2-dev \
    libopenal-dev \
    libfreetype-dev \
    liblua5.1-0-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY OpenRA /src/openra
WORKDIR /src/openra

# Fix Windows CRLF line endings in shell scripts (git autocrlf on Windows adds \r)
RUN find . -name '*.sh' -exec sed -i 's/\r$//' {} + && \
    find . -name '*.sh' -exec chmod +x {} +

# Build with system libraries (unix-generic avoids bundled native binaries)
# SKIP_PROTOC=true uses pre-generated protobuf C# files (avoids protoc arm64 crash in Docker)
ENV SKIP_PROTOC=true
RUN make TARGETPLATFORM=unix-generic CONFIGURATION=Release

# Verify critical output (includes Null platform for headless RL operation)
RUN test -f bin/OpenRA.dll && \
    test -f bin/OpenRA.Game.dll && \
    test -f bin/OpenRA.Mods.Common.dll && \
    test -f bin/OpenRA.Platforms.Null.dll

# ==============================================================================
# Stage 2: Install Python dependencies
# ==============================================================================
FROM python:3.11-slim-bookworm AS python-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/
COPY openra_env/ /app/openra_env/
COPY proto/ /app/proto/
COPY README.md /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# ==============================================================================
# Stage 3: Runtime image
# ==============================================================================
FROM mcr.microsoft.com/dotnet/aspnet:8.0-bookworm-slim AS dotnet-runtime

FROM python:3.11-slim-bookworm

LABEL maintainer="OpenRA-RL"
LABEL description="OpenRA RL Environment - headless game engine with gRPC bridge + OpenEnv API"

# Copy ASP.NET Core runtime from official Microsoft image
COPY --from=dotnet-runtime /usr/share/dotnet /usr/share/dotnet
RUN ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libegl-mesa0 \
    mesa-vulkan-drivers \
    libvulkan1 \
    libsdl2-2.0-0 \
    libopenal1 \
    libfreetype6 \
    liblua5.1-0 \
    libicu72 \
    curl procps \
    x11vnc novnc websockify \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=python-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-build /usr/local/bin /usr/local/bin

# Copy built OpenRA (bin, mods, glsl shaders, and global mix database for content resolution)
COPY --from=openra-build /src/openra/bin /opt/openra/bin
COPY --from=openra-build /src/openra/mods /opt/openra/mods
COPY --from=openra-build /src/openra/glsl /opt/openra/glsl
COPY --from=openra-build ["/src/openra/global mix database.dat", "/opt/openra/global mix database.dat"]

# Create native library symlinks that OpenRA expects
# (configure-system-libraries.sh points these to system lib paths)
RUN LIBDIR=$( [ "$(dpkg --print-architecture)" = "arm64" ] && echo "/usr/lib/aarch64-linux-gnu" || echo "/usr/lib/x86_64-linux-gnu" ) && \
    ln -sf "$LIBDIR/libSDL2-2.0.so.0" /opt/openra/bin/SDL2.so && \
    ln -sf "$LIBDIR/libopenal.so.1" /opt/openra/bin/soft_oal.so && \
    ln -sf "$LIBDIR/libfreetype.so.6" /opt/openra/bin/freetype6.so && \
    ln -sf "$LIBDIR/liblua5.1.so.0" /opt/openra/bin/lua51.so

# Copy Python application code
COPY openra_env/ /app/openra_env/
COPY proto/ /app/proto/
COPY pyproject.toml /app/

# Create OpenRA support directory and pre-install RA game content
# (required for replay viewer which uses Game.Platform=Default with full UI)
RUN mkdir -p /root/.config/openra/Content/ra/v2/expand /root/.config/openra/Content/ra/v2/cnc && \
    curl -sL -o /tmp/ra-quickinstall.zip \
        https://openra.baxxster.no/openra/ra-quickinstall.zip && \
    apt-get update && apt-get install -y --no-install-recommends unzip && \
    unzip -o /tmp/ra-quickinstall.zip -d /tmp/ra-content && \
    cp /tmp/ra-content/*.mix /root/.config/openra/Content/ra/v2/ && \
    cp /tmp/ra-content/expand/* /root/.config/openra/Content/ra/v2/expand/ && \
    cp /tmp/ra-content/cnc/* /root/.config/openra/Content/ra/v2/cnc/ && \
    rm -rf /tmp/ra-quickinstall.zip /tmp/ra-content && \
    apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy entrypoints (fix Windows CRLF line endings)
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/replay-viewer.sh /replay-viewer.sh
RUN sed -i 's/\r$//' /entrypoint.sh /replay-viewer.sh && \
    chmod +x /entrypoint.sh /replay-viewer.sh

# Environment
ENV OPENRA_PATH=/opt/openra
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1
ENV DOTNET_ROLL_FORWARD=LatestMajor
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
# Game configuration (override at runtime with -e)
ENV AI_SLOT=Multi0
ENV BOT_TYPE=normal
ENV RECORD_REPLAYS=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "openra_env.server.app"]
