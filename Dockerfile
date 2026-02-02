# Build stage - use trixie for glibc 2.38+ (required by ort)
FROM rust:1.93-trixie AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml ./

# Create a dummy main.rs to cache dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs

# Build dependencies only
RUN cargo build --release && rm -rf src

# Copy actual source code
COPY src ./src

# Build the application
RUN touch src/main.rs && cargo build --release

# Runtime stage - use trixie for glibc 2.38+ compatibility
FROM debian:trixie-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /app/target/release/wazo-stt-server /app/wazo-stt-server

# Create models directory
RUN mkdir -p /models/parakeet

# Download Parakeet model script
COPY scripts/download-model.sh /app/scripts/download-model.sh
RUN chmod +x /app/scripts/download-model.sh

# Environment variables
ENV MODEL_PATH=/models/parakeet
ENV HOST=0.0.0.0
ENV PORT=8000
ENV RUST_LOG=info

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["/app/wazo-stt-server"]
