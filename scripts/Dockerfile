
FROM archlinux:latest
RUN pacman --noconfirm -Syu --noconfirm llvm clang cmake make diffutils wget unzip

# Downlaod c2ffi
RUN wget https://github.com/rpav/c2ffi/archive/refs/heads/llvm-18.1.0.zip && \
    unzip llvm-18.1.0.zip && \
    mv c2ffi-llvm-18.1.0 /c2ffi && \
    rm llvm-18.1.0.zip

# Build c2ffi
RUN cd /c2ffi && \
        rm -rf build && mkdir -p build && cd build && \
        cmake -DBUILD_CONFIG=Release .. && make

WORKDIR /data
ENTRYPOINT ["/c2ffi/build/bin/c2ffi"]
