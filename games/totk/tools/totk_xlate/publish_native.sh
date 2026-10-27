#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
BUILD="${TOTK_XLATE_BUILD:-$REPO/games/totk/build/totk_xlate}"
python3 - "$REPO" "$BUILD" "${SPIRV_CROSS_SRC:-}" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from build_app import GAME, build_shader_service

build_shader_service(GAME, GAME/'build/shader-service-logs', build=Path(sys.argv[2]),
                     cross=Path(sys.argv[3]) if sys.argv[3] else None)
PY
cmake --build "$BUILD" -j 4
ctest --test-dir "$BUILD" --output-on-failure
