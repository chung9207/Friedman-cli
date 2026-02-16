# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
