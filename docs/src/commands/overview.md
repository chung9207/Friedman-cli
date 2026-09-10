# CLI Reference Overview

Friedman-cli uses an **action-first** command hierarchy: commands are organized by action (`estimate`, `irf`, `forecast`, ...) rather than by model type.

<!-- BEGIN GENERATED: do not hand-edit; run julia docs/generate_cli_reference.jl -->

## Command Tree

```
friedman
├── completions     bash | fish | zsh
├── data     balance | describe | diagnose | dropna | export | filter | fix | import | keeprows | list | load | transform | validate
├── did     estimate | event-study | lp-did | test bacon | test honest | test negweight | test pretrend
├── dsge     bank irf | bank pe | bank steady-state | bank transition | bayes compare | bayes estimate | bayes fevd | bayes hd | bayes identification | bayes irf | bayes learning-rate | bayes marginal-lik | bayes mcmc-diag | bayes overlap | bayes posterior-mode | bayes predictive | bayes prior-predictive | bayes simulate | bayes summary | ct fevd | ct irf | ct solve | ct transition | dcegm fevd | dcegm irf | dcegm simulate | dcegm solve | dcegm steady-state | dcegm transition | determinacy-map | estimate | fevd | firm irf | firm steady-state | firm transition | ha accuracy | ha distribution-irf | ha estimate | ha fevd | ha hd | ha inequality-irf | ha irf | ha simulate | ha simulate-panel | ha solve | ha steady-state | hd | irf | lifecycle fevd | lifecycle irf | lifecycle simulate | lifecycle steady-state | lifecycle transition | moments | olg fevd | olg irf | olg simulate | olg solve | perfect-foresight | simulate | solve | steady-state
├── estimate     3sls | aparch | arch | ardl | arfima | arima | bekk | bvar | ccc | cgarch | cointreg | dcc | dynamic | egarch | elastic-net | fastica | favar | fiegarch | figarch | garch | garch-midas | gdfm | gjr-garch | gmm | heckman | igarch | iv | kde | kernel-reg | lasso | logit | lowess | lp | mfvar | midas | ml | mlogit | ms | ms-ar | nardl | nbreg | ologit | oprobit | piv | plogit | pmg | poisson | pprobit | preg | probit | pvar | qreg | rdd | reg | ridge | robust | sarima | sdfm | select | setar | smm | star | statespace | static | sur | sv | svar | svec | threshold | tobit | truncreg | tvp | tvpvar | var | vecm | xtcointreg
├── fevd     bvar | favar | lp | pvar | sdfm | var | vecm
├── filter     bhp | bk | bn | hamilton | hp | x13
├── forecast     aparch | arch | arfima | arima | bvar | cgarch | dynamic | egarch | evaluate clark-west | evaluate combine | evaluate dm | evaluate encompassing | evaluate metrics | evaluate mincer-zarnowitz | favar | fiegarch | figarch | garch | garch-midas | gdfm | gjr-garch | igarch | lp | midas | ms | ms-ar | sarima | scenario | sdfm | setar | star | static | sv | var | vecm
├── hd     bvar | favar | lp | var | vecm
├── io     aggregate | balance | baqaee-farhi | bf elasticities | bf equilibrium | bf local | bf misallocation | bf network | bf shock-curve | bf wedges | bilateral-trade | download | export-decomposition | extract | footprint | ghosh | impact | key-sectors | leontief | linkages | load | multipliers | network-stats | price | sda | sources | vertical-specialization
├── irf     bvar | favar | lp | pvar | sdfm | tvpvar | var | vecm
├── model     info | reproduce
├── multipliers     nardl
├── nowcast     bridge | bvar | dfm | forecast | news
├── policy     counterfactual bvar | counterfactual lp | counterfactual var | effects bvar | effects lp | effects sign | effects var | history bvar | history var | jacobian ha | moments bvar | moments var | news dsge | news ha | opp bvar | opp var | opp-sequence bvar | opp-sequence var | optimal bvar | optimal lp | optimal var | spanning var | sufficiency dsge
├── predict     3sls | aparch | arch | arfima | arima | bvar | cgarch | dynamic | egarch | favar | fiegarch | figarch | garch | garch-midas | gdfm | gjr-garch | igarch | logit | mlogit | ms | ms-ar | nbreg | ologit | oprobit | piv | plogit | poisson | pprobit | preg | probit | reg | sarima | statespace | static | sur | sv | var | vecm
├── residuals     3sls | aparch | arch | arfima | arima | bvar | cgarch | dynamic | egarch | favar | fiegarch | figarch | garch | garch-midas | gdfm | gjr-garch | igarch | logit | mlogit | ms | ms-ar | nbreg | ologit | oprobit | piv | plogit | poisson | pprobit | preg | probit | reg | sarima | setar | star | statespace | static | sur | sv | var | vecm
├── serve
├── show
├── spectral     acf | cross | density | periodogram | transfer
└── test     adf | adf-2break | anderson-rubin | andrews | arch-lm | ardl-bounds | bai-perron | bartlett-wn | bds | box-pierce | brant | breitung | breusch-pagan | chow | cips | cusum | cusumsq | dfgls | dh-causality | dispersion | durbin-watson | edf | engle-granger | ers | f-fe | factor-break | fisher | fisher-johansen | fourier-adf | fourier-kpss | glejser | gph | granger | gregory-hansen | gsadf | hadri | hansen-instability | hansen-linearity | harvey | hausman | hausman-iia | hegy | heteroskedasticity | identifiability | influence | ips | johansen | kao | kpss | ljung-box | llc | lm | lm-unitroot | local-whittle | lr | modified-wald | moon-perron | nardl-symmetry | normality | np | nyblom | panic | park-added | pedroni | pesaran-cd | phillips-ouliaris | pmg-hausman | pp | pvar hansen-j | pvar lagselect | pvar mmsc | pvar stability | recursive-residuals | sadf | sign-bias | star-linearity | var lagselect | var stability | variance-ratio | vecm alpha | vecm beta | vecm joint | vecm known-beta | vecm weak-exog | vif | weak-instrument | westerlund | white | wild-cluster | wooldridge-ar | za

Total: 21 top-level commands, 456 leaves (from registry).
```

Additionally, `friedman repl` launches an interactive REPL session with persistent data loading, result caching, and tab completion.

## Generated reference pages

- [`completions`](generated/completions.md) — 3 leaves
- [`data`](generated/data.md) — 13 leaves
- [`did`](generated/did.md) — 7 leaves
- [`dsge`](generated/dsge.md) — 62 leaves
- [`estimate`](generated/estimate.md) — 76 leaves
- [`fevd`](generated/fevd.md) — 7 leaves
- [`filter`](generated/filter.md) — 6 leaves
- [`forecast`](generated/forecast.md) — 35 leaves
- [`hd`](generated/hd.md) — 5 leaves
- [`io`](generated/io.md) — 27 leaves
- [`irf`](generated/irf.md) — 8 leaves
- [`model`](generated/model.md) — 2 leaves
- [`multipliers`](generated/multipliers.md) — 1 leaves
- [`nowcast`](generated/nowcast.md) — 5 leaves
- [`policy`](generated/policy.md) — 23 leaves
- [`predict`](generated/predict.md) — 38 leaves
- [`residuals`](generated/residuals.md) — 40 leaves
- [`serve`](generated/serve.md) — 1 leaves
- [`show`](generated/show.md) — 1 leaves
- [`spectral`](generated/spectral.md) — 5 leaves
- [`test`](generated/test.md) — 91 leaves

<!-- END GENERATED -->

## Common Options

All commands that produce output support these options:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | Output format: `table`, `csv`, or `json` |
| `--output` | `-o` | String | (stdout) | Export results to a file path |

## Help

Every command and subcommand supports `--help`. Machine-readable schema:

```bash
friedman schema estimate var
```
