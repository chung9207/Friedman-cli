#compdef friedman
# Friedman-cli zsh completion (generated)
_friedman() {
  local -a commands
  commands=(
    'completions'
    'data'
    'did'
    'dsge'
    'estimate'
    'fevd'
    'filter'
    'forecast'
    'hd'
    'io'
    'irf'
    'model'
    'multipliers'
    'nowcast'
    'policy'
    'predict'
    'residuals'
    'serve'
    'show'
    'spectral'
    'test'
    'repl'
  )
  _describe 'command' commands
}
_friedman "$@"
