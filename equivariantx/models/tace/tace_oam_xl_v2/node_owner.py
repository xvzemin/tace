"""Separate source-owned mixed reductions from edge-owned radial derivatives."""
from functools import lru_cache
from equivariantx.utils.metadata import parse_metadata


@lru_cache(maxsize=128)
def split_source_program(program):
    terms=parse_metadata(program)
    result=[]
    for selected in ({0},{1,3,4}):
        slots=sorted({slot for _,_,pairs in terms for role,slot in pairs if role in selected})
        if not slots:continue
        remap={slot:i for i,slot in enumerate(slots)}
        sub=tuple((mapping,weighted,tuple((role,remap[slot]) for role,slot in pairs if role in selected))
                  for mapping,weighted,pairs in terms if any(role in selected for role,_ in pairs))
        result.append((selected=={0},repr(sub),slots))
    return result


def source_owned_edges(metadata,program,source,target,operands,asynchronous):
    from .receiver import receiver_edges
    from .edge import run_edges
    results={}
    for is_node,sub,slots in split_source_program(program):
        output=(receiver_edges(metadata,sub,source,target,operands,asynchronous) if is_node
                else run_edges(metadata,sub,source,target,operands))
        results.update(zip(slots,output))
    return [results[i] for i in range(len(results))]
