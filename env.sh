# On NixOS the manylinux wheels (torch, onnxruntime) look for libstdc++
# and zlib on the library path, where they are not.
#   source env.sh
libs=$(nix build --no-link --print-out-paths \
       nixpkgs#stdenv.cc.cc.lib nixpkgs#zlib 2>/dev/null | sed 's|$|/lib|' | paste -sd:)
if [ -n "$libs" ]; then
  export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  echo "env.sh: nix unavailable, LD_LIBRARY_PATH left unchanged" >&2
fi
