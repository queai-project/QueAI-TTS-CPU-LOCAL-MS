#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   bash clean_voices.sh                 # simulación, no borra nada
#   bash clean_voices.sh --apply         # aplica cambios
#   bash clean_voices.sh /ruta/voices    # simulación sobre otra ruta
#   bash clean_voices.sh /ruta/voices --apply

VOICES_DIR="voices"
APPLY=0

for arg in "$@"; do
  case "$arg" in
    --apply)
      APPLY=1
      ;;
    *)
      VOICES_DIR="$arg"
      ;;
  esac
done

if [[ ! -d "$VOICES_DIR" ]]; then
  echo "Error: no existe la carpeta '$VOICES_DIR'"
  exit 1
fi

echo "Carpeta objetivo: $VOICES_DIR"
echo "Tamaño antes:"
du -sh "$VOICES_DIR"

echo
echo "Se conservará:"
echo "  - $VOICES_DIR/_script/**"
echo "  - cualquier carpeta samples/**"
echo "  - cualquier archivo *.onnx.json"

echo
echo "Archivos que se eliminarían:"
find "$VOICES_DIR" -type f \
  ! -path "$VOICES_DIR/_script/*" \
  ! -path "*/samples/*" \
  ! -name "*.onnx.json" \
  -print

echo
echo "Directorios vacíos que quedarían y se podrían borrar:"
find "$VOICES_DIR" -type d -empty \
  ! -path "$VOICES_DIR/_script" \
  ! -path "*/samples" \
  -print

if [[ "$APPLY" -eq 0 ]]; then
  echo
  echo "Simulación completada. No se borró nada."
  echo "Para aplicar de verdad:"
  echo "  bash $0 \"$VOICES_DIR\" --apply"
  exit 0
fi

echo
echo "Aplicando borrado..."

# Borra archivos no permitidos
find "$VOICES_DIR" -type f \
  ! -path "$VOICES_DIR/_script/*" \
  ! -path "*/samples/*" \
  ! -name "*.onnx.json" \
  -delete

# Borra directorios vacíos repetidamente, por si quedan anidados
while true; do
  EMPTY_COUNT="$(find "$VOICES_DIR" -type d -empty \
    ! -path "$VOICES_DIR/_script" \
    ! -path "*/samples" | wc -l)"
  [[ "$EMPTY_COUNT" -eq 0 ]] && break

  find "$VOICES_DIR" -type d -empty \
    ! -path "$VOICES_DIR/_script" \
    ! -path "*/samples" \
    -delete
done

echo
echo "Limpieza terminada."
echo "Tamaño después:"
du -sh "$VOICES_DIR"