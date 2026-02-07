# Rename command: rename stored tags

function register_rename_commands!()
    return LeafCommand("rename", _rename;
        args=[
            Argument("old_tag"; description="Current tag name (e.g. var001)"),
            Argument("new_tag"; description="New tag name"),
        ],
        options=Option[],
        description="Rename a stored model or result tag")
end

function _rename(; old_tag::String, new_tag::String)
    success = storage_rename!(old_tag, new_tag)
    if success
        printstyled("Renamed: $old_tag -> $new_tag\n"; color=:green)
    else
        printstyled("Tag '$old_tag' not found\n"; color=:red)
    end
end
