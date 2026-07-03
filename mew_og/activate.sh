#!/usr/bin/env bash

# Source this file before running MEW-OG commands:
#   source activate.sh
#
# The script creates a local .venv on first use and installs the package from
# pyproject.toml. Set MEW_OG_PYTHON=/path/to/python to choose a specific Python.

_MEW_OG_ERREXIT_WAS_SET=0
case "$-" in
  *e*) _MEW_OG_ERREXIT_WAS_SET=1 ;;
esac
set -e

MEW_OG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MEW_OG_VENV_DIR:-${MEW_OG_ROOT}/.venv}"
STAMP_FILE="${VENV_DIR}/.mew_og_install_stamp"
PYPROJECT="${MEW_OG_ROOT}/pyproject.toml"

function _mew_og_is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

function _mew_og_find_python() {
  if [[ -n "${MEW_OG_PYTHON:-}" ]]; then
    echo "${MEW_OG_PYTHON}"
    return
  fi

  for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return
    fi
  done

  return 1
}

function _mew_og_python_ok() {
  local python_bin="$1"
  "${python_bin}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
}

function _mew_og_create_venv() {
  local python_bin
  python_bin="$(_mew_og_find_python)" || {
    echo "No Python >=3.9 candidate found. Set MEW_OG_PYTHON=/path/to/python." >&2
    return 1
  }

  if ! _mew_og_python_ok "${python_bin}"; then
    echo "Python '${python_bin}' is below MEW-OG's requirement (>=3.9)." >&2
    echo "Set MEW_OG_PYTHON=/path/to/python3.9+ and source this script again." >&2
    return 1
  fi

  echo "Creating MEW-OG virtual environment at ${VENV_DIR}"
  "${python_bin}" -m venv "${VENV_DIR}"
}

function _mew_og_install_requirements() {
  echo "Installing MEW-OG dependencies from ${PYPROJECT}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip

  if [[ "${MEW_OG_INSTALL_DEV:-0}" == "1" ]]; then
    "${VENV_DIR}/bin/python" -m pip install -e "${MEW_OG_ROOT}[dev]"
  else
    "${VENV_DIR}/bin/python" -m pip install -e "${MEW_OG_ROOT}"
  fi

  touch "${STAMP_FILE}"
}

if [[ ! -d "${VENV_DIR}" ]]; then
  _mew_og_create_venv
fi

if [[ ! -f "${STAMP_FILE}" || "${PYPROJECT}" -nt "${STAMP_FILE}" ]]; then
  _mew_og_install_requirements
fi

if _mew_og_is_sourced; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "MEW-OG environment is ready at ${VENV_DIR}"
  echo "Run: source ${MEW_OG_ROOT}/activate.sh"
fi

export MEW_OG_ROOT
export PYTHONPATH="${MEW_OG_ROOT}:${PYTHONPATH:-}"

echo
echo "*************************************"

if [[ "${_MEW_OG_ERREXIT_WAS_SET}" == "0" ]]; then
  set +e
fi
unset _MEW_OG_ERREXIT_WAS_SET
echo "*       ACTIVATED MEW-OG ENV        *"
echo "*************************************"
printf "* %-15s: %s\n" "MEW_OG_ROOT" "${MEW_OG_ROOT}"
printf "* %-15s: %s\n" "VENV_DIR" "${VENV_DIR}"
printf "* %-15s: %s\n" "PYTHON" "$("${VENV_DIR}/bin/python" -V)"
echo "*************************************"
