# Sous NixOS, les wheels manylinux (torch, onnxruntime) cherchent libstdc++
# et zlib dans le chemin des bibliotheques, ou elles ne sont pas.
#   source env.sh
libs=$(nix build --no-link --print-out-paths \
       nixpkgs#stdenv.cc.cc.lib nixpkgs#zlib 2>/dev/null | sed 's|$|/lib|' | paste -sd:)
if [ -n "$libs" ]; then
  export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  echo "env.sh : nix indisponible, LD_LIBRARY_PATH inchange" >&2
fi
