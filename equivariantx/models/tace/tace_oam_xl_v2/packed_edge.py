"""Physical irrep blocks shared by convolution and its following GEMMs.

The logical tensor shape is unchanged. Every read and write of role four,
including recursive mixed adjoints, uses the same physical address map.
"""

from functools import lru_cache
import re

from equivariantx.utils.metadata import parse_metadata


@lru_cache(maxsize=128)
def pack_edge_metadata(metadata):
    """Describe consecutive equal-degree UVU paths without changing them."""
    paths, width, weighted = parse_metadata(metadata)
    if weighted or not paths or any(p[4] != 1 for p in paths):
        raise ValueError("Packed edge outputs require explicit UVU paths")
    blocks = []
    end = 0
    for path in sorted(paths, key=lambda p: p[2]):
        start, channels, degree = path[2], path[3], path[7]
        if start != end:
            raise ValueError("Packed outputs require disjoint contiguous paths")
        if blocks and blocks[-1][1:3] == (channels, degree):
            blocks[-1] = (*blocks[-1][:3], blocks[-1][3] + 1)
        else:
            blocks.append((start, channels, degree, 1))
        end += channels * degree
    return repr((paths, width, weighted, ("packed_output", tuple(blocks))))


def packed_description(metadata):
    """Return the optional physical layout descriptor."""
    description = parse_metadata(metadata)
    if len(description) == 4 and description[3][0] == "packed_output":
        return description[3][1]
    return None


def packed_address(blocks, row, column):
    """Map a statically known angular column and one channel lane."""
    for start, channels, degree, paths in blocks:
        if start <= column < start + channels * degree * paths:
            offset = column - start
            if offset % channels:
                raise ValueError("Expected a channel-aligned angular column")
            path, magnetic = divmod(offset // channels, degree)
            total_channels = channels * paths
            return (f"packed_nodes*{start}+({row})*{total_channels*degree}"
                    f"+{magnetic*total_channels+path*channels}+u")
    raise ValueError(f"Output column {column} is outside the packed layout")


def pack_edge_source(source, blocks, program):
    """Apply a checked address transform to generated native CUDA source.

    All native schedules retain their existing reduction ownership and path
    grouping. The node count is an extra runtime scalar, so one compiled
    kernel supports every batch capacity.
    """
    input_roles, output_roles = {}, {}
    for mapping, _, pairs in program:
        for role, operand in enumerate(mapping):
            input_roles.setdefault(operand, set()).add(role)
        for role, slot in pairs:
            output_roles.setdefault(slot, set()).add(role)
    for roles in (*input_roles.values(), *output_roles.values()):
        if 4 in roles and roles != {4}:
            raise ValueError("Packed and unpacked operands cannot alias")
    pattern = re.compile(
        r"(?P<kind>[pg])(?P<slot>\d+)(?P<operator>\[|\s*\+\s*)"
        r"\(?(?P<row>target_node|source_node|node|row)\)?\s*\*\s*\d+"
        r"\s*\+\s*(?P<column>\d+)\s*\+\s*u"
    )
    replacements = 0

    def replace(match):
        nonlocal replacements
        kind, slot = match['kind'], int(match['slot'])
        roles = (input_roles if kind == 'p' else output_roles).get(slot, set())
        if roles != {4}:
            return match[0]
        replacements += 1
        address = packed_address(blocks, match['row'], int(match['column']))
        operator = '[' if match['operator'] == '[' else ' + '
        return f"{kind}{slot}{operator}({address})"

    source = pattern.sub(replace, source)
    # A phase may exclusively compute angular/radial outputs but still reads
    # role four. If it neither reads nor writes that role, the argument remains
    # harmless and all launch signatures still agree.
    source = re.sub(r'(void (?:run|phase\d+)\([^)]*)(\)\s*\{)',
                    r'\1, long long packed_nodes\2', source)
    source = re.sub(r'(case \d+: phase\d+\([^;]+)(\); break;)',
                    r'\1, packed_nodes\2', source)
    if 'packed_nodes' not in source:
        raise ValueError("No native CUDA entry point was transformed")
    return source
