# API Reference

Internal API documentation for the Friedman module. Most symbols are internal (prefixed with `_`) and not exported. Only [`main`](@ref) and [`build_app`](@ref) are exported.

## Exported Functions

```@docs
Friedman.main
Friedman.build_app
```

## CLI Types

```@docs
Friedman.Argument
Friedman.Option
Friedman.Flag
Friedman.LeafCommand
Friedman.NodeCommand
Friedman.Entry
```

## CLI Parser

```@docs
Friedman.tokenize
Friedman.resolve_option
Friedman.resolve_flag
Friedman.convert_value
Friedman.bind_args
```

## CLI Dispatch

```@docs
Friedman.dispatch
Friedman.dispatch_node
Friedman.dispatch_leaf
```

## CLI Help

```@docs
Friedman.print_help
Friedman.print_entry_line
```

## Data I/O

```@docs
Friedman.load_data
Friedman.df_to_matrix
Friedman.variable_names
Friedman.output_result
Friedman.output_kv
```

## Configuration

```@docs
Friedman.load_config
Friedman.get_identification
Friedman.get_prior
Friedman.get_gmm
Friedman.get_nongaussian
```

## Storage

```@docs
Friedman._storage_path
Friedman._load_storage
Friedman._save_storage!
Friedman.auto_tag
Friedman.serialize_model
Friedman.storage_save!
Friedman.storage_save_auto!
Friedman.storage_load
Friedman.storage_list
Friedman.storage_rename!
Friedman.resolve_stored_tags
```

## Settings

```@docs
Friedman.friedman_home
Friedman.ensure_friedman_home!
Friedman.get_username
Friedman.init_settings!
Friedman.load_settings
Friedman.save_settings!
Friedman.load_projects
Friedman.save_projects!
Friedman.register_project!
```

## Shared Utilities

```@docs
Friedman.ID_METHOD_MAP
Friedman._load_and_estimate_var
Friedman._load_and_estimate_bvar
Friedman._build_prior
Friedman._build_check_func
Friedman._build_identification_kwargs
Friedman._load_and_structural_lp
Friedman._var_forecast_point
```
