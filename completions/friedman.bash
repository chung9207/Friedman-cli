# Friedman-cli bash completion (generated)
_friedman() {
  local cur prev words cword
  _init_completion || return
  if [[ $cword -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "completions data did dsge estimate fevd filter forecast hd io irf model multipliers nowcast policy predict residuals serve show spectral test repl" -- "$cur") )
    return
  fi
  case "${words[1]}" in
    completions) COMPREPLY=( $(compgen -W "bash fish zsh" -- "$cur") ) ;;
    data) COMPREPLY=( $(compgen -W "balance describe diagnose dropna export filter fix import keeprows list load transform validate" -- "$cur") ) ;;
    did) COMPREPLY=( $(compgen -W "estimate event-study lp-did test" -- "$cur") ) ;;
    dsge) COMPREPLY=( $(compgen -W "bank bayes ct dcegm determinacy-map estimate fevd firm ha hd irf lifecycle moments olg perfect-foresight simulate solve steady-state" -- "$cur") ) ;;
    estimate) COMPREPLY=( $(compgen -W "3sls aparch arch ardl arfima arima bekk bvar ccc cgarch cointreg dcc dynamic egarch elastic-net fastica favar fiegarch figarch garch garch-midas gdfm gjr-garch gmm heckman igarch iv kde kernel-reg lasso logit lowess lp mfvar midas ml mlogit ms ms-ar nardl nbreg ologit oprobit piv plogit pmg poisson pprobit preg probit pvar qreg rdd reg ridge robust sarima sdfm select setar smm star statespace static sur sv svar svec threshold tobit truncreg tvp tvpvar var vecm xtcointreg" -- "$cur") ) ;;
    fevd) COMPREPLY=( $(compgen -W "bvar favar lp pvar sdfm var vecm" -- "$cur") ) ;;
    filter) COMPREPLY=( $(compgen -W "bhp bk bn hamilton hp x13" -- "$cur") ) ;;
    forecast) COMPREPLY=( $(compgen -W "aparch arch arfima arima bvar cgarch dynamic egarch evaluate favar fiegarch figarch garch garch-midas gdfm gjr-garch igarch lp midas ms ms-ar sarima scenario sdfm setar star static sv var vecm" -- "$cur") ) ;;
    hd) COMPREPLY=( $(compgen -W "bvar favar lp var vecm" -- "$cur") ) ;;
    io) COMPREPLY=( $(compgen -W "aggregate balance baqaee-farhi bf bilateral-trade download export-decomposition extract footprint ghosh impact key-sectors leontief linkages load multipliers network-stats price sda sources vertical-specialization" -- "$cur") ) ;;
    irf) COMPREPLY=( $(compgen -W "bvar favar lp pvar sdfm tvpvar var vecm" -- "$cur") ) ;;
    model) COMPREPLY=( $(compgen -W "info reproduce" -- "$cur") ) ;;
    multipliers) COMPREPLY=( $(compgen -W "nardl" -- "$cur") ) ;;
    nowcast) COMPREPLY=( $(compgen -W "bridge bvar dfm forecast news" -- "$cur") ) ;;
    policy) COMPREPLY=( $(compgen -W "counterfactual effects history jacobian moments news opp opp-sequence optimal spanning sufficiency" -- "$cur") ) ;;
    predict) COMPREPLY=( $(compgen -W "3sls aparch arch arfima arima bvar cgarch dynamic egarch favar fiegarch figarch garch garch-midas gdfm gjr-garch igarch logit mlogit ms ms-ar nbreg ologit oprobit piv plogit poisson pprobit preg probit reg sarima statespace static sur sv var vecm" -- "$cur") ) ;;
    residuals) COMPREPLY=( $(compgen -W "3sls aparch arch arfima arima bvar cgarch dynamic egarch favar fiegarch figarch garch garch-midas gdfm gjr-garch igarch logit mlogit ms ms-ar nbreg ologit oprobit piv plogit poisson pprobit preg probit reg sarima setar star statespace static sur sv var vecm" -- "$cur") ) ;;
    serve) COMPREPLY=( $(compgen -W "" -- "$cur") ) ;;
    show) COMPREPLY=( $(compgen -W "" -- "$cur") ) ;;
    spectral) COMPREPLY=( $(compgen -W "acf cross density periodogram transfer" -- "$cur") ) ;;
    test) COMPREPLY=( $(compgen -W "adf adf-2break anderson-rubin andrews arch-lm ardl-bounds bai-perron bartlett-wn bds box-pierce brant breitung breusch-pagan chow cips cusum cusumsq dfgls dh-causality dispersion durbin-watson edf engle-granger ers f-fe factor-break fisher fisher-johansen fourier-adf fourier-kpss glejser gph granger gregory-hansen gsadf hadri hansen-instability hansen-linearity harvey hausman hausman-iia hegy heteroskedasticity identifiability influence ips johansen kao kpss ljung-box llc lm lm-unitroot local-whittle lr modified-wald moon-perron nardl-symmetry normality np nyblom panic park-added pedroni pesaran-cd phillips-ouliaris pmg-hausman pp pvar recursive-residuals sadf sign-bias star-linearity var variance-ratio vecm vif weak-instrument westerlund white wild-cluster wooldridge-ar za" -- "$cur") ) ;;
  esac
}
complete -F _friedman friedman
