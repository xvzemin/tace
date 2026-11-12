"""Native reciprocal radial rows and exact recursively generated adjoints."""
from functools import lru_cache
import re
import os
import torch

from equivariantx.ops.contraction import adjoint_program
from equivariantx.utils.autograd import gradient_mask
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs


@lru_cache(maxsize=256)
def compressed_source(source,program):
    """Redirect radial reads and reduce repeated-row adjoints inside CG kernels."""
    input_roles,output_roles={},{}
    for mapping,_,pairs in program:
        for role,pointer in enumerate(mapping):input_roles.setdefault(pointer,set()).add(role)
        for role,slot in pairs:output_roles.setdefault(slot,set()).add(role)
    if any(1 in roles and roles!={1} for roles in (*input_roles.values(),*output_roles.values())):
        raise ValueError('Radial and nonradial buffers cannot alias')
    def address(match):
        kind,slot,expr=match['kind'],int(match['slot']),match['expr']
        roles=(input_roles if kind=='p' else output_roles).get(slot,set())
        if roles=={1}:expr=re.sub(r'\bedge\b','radial_index[edge]',expr)
        return f'{kind}{slot}[{expr}]'
    source=re.sub(r'(?P<kind>[pg])(?P<slot>\d+)\[(?P<expr>[^\]\n]+)\]',address,source)
    lines=[]
    for line in source.splitlines():
        match=re.fullmatch(r'if \(active\) (g(\d+)\[.+\]) (?:\+=|=) (\w+);',line.strip())
        if match and output_roles.get(int(match[2]))=={1}:
            line=f'if (active) atomicAdd(&{match[1]}, {match[3]});'
        pointer_store=re.fullmatch(r'if \(active\) \*\((g(\d+) [^;]+)\) = (\w+);',line.strip())
        if pointer_store and output_roles.get(int(pointer_store[2]))=={1}:
            address=re.sub(r'\bedge\b','radial_index[edge]',pointer_store[1])
            line=f'if (active) atomicAdd({address}, {pointer_store[3]});'
        lines.append(line)
    source='\n'.join(lines)
    source=re.sub(r'(void (?:run|phase\d+)\([^)]*)(\)\s*\{)',
                  r'\1, const int64_t* radial_index\2',source)
    source=re.sub(r'(case \d+: phase\d+\([^;]+)(\); break;)',r'\1, radial_index\2',source)
    return source


def run_compressed(metadata,program,source,target,operands,radial_map):
    """Evaluate every explicit term without expanding the radial edge arrays."""
    from .cg_convolution import contract_direct
    terms=parse_metadata(program)
    roles={role for _,_,pairs in terms for role,_ in pairs}
    if os.environ.get('EQX_XL_COMPRESSED_SCHEDULED','1')=='1':
        from .edge import run_edges, RECEIVER_PIPELINE
        if source.numel() >= 1024 and roles in ({0,1,3},{1,3}):
            return run_edges(metadata,program,source,target,operands,radial_map=radial_map)
        paths=parse_metadata(metadata)[0]
        if (RECEIVER_PIPELINE!='off' and roles=={4} and paths
                and all(p[3]==128 and p[4]==1 and p[8]>=0 for p in paths)):
            from .receiver import receiver_edges
            asynchronous=RECEIVER_PIPELINE=='async' or (RECEIVER_PIPELINE=='auto' and len(terms)==1)
            return receiver_edges(metadata,program,source,target,operands,asynchronous,radial_map)
    if (os.environ.get('EQX_XL_RADIAL_OWNER','0')=='1' and 1 in roles
            and all(operands[mapping[1]].shape[0]!=1 for mapping,_,_ in terms)):
        from .radial_owner import radial_owned_edges
        return radial_owned_edges(metadata,program,source,target,operands,radial_map)
    outputs=o3_outputs(metadata,program,source,target,operands)
    for output in outputs:output.zero_()
    if not source.numel():return outputs
    required={index for mapping,_,pairs in terms for output,_ in pairs
              for role,index in enumerate(mapping) if role!=output}
    operands=[x.contiguous() if i in required else x for i,x in enumerate(operands)]
    if any(operands[mapping[2]].numel() for mapping,_,_ in terms):
        raise ValueError('Compressed radial heads must be materialized')
    calls=[(tuple(role for role,_ in pairs),tuple(operands[i] for i in mapping),
            tuple(outputs[slot] for _,slot in pairs),weighted)
           for mapping,weighted,pairs in terms]
    contract_direct(metadata,source.contiguous(),target.contiguous(),calls,
                    batch_paths=True,radial_map=radial_map.contiguous())
    return outputs


@torch.library.custom_op('eqx_oam_xl::compressed_edge',mutates_args=(),device_types='cuda')
def compressed_edge(metadata:str,program:str,source:torch.Tensor,target:torch.Tensor,
                    operands:list[torch.Tensor],radial_map:torch.Tensor)->list[torch.Tensor]:
    return run_compressed(metadata,program,source,target,operands,radial_map)


@compressed_edge.register_fake
def fake_compressed(metadata,program,source,target,operands,radial_map):
    return o3_outputs(metadata,program,source,target,operands)


def compressed_context(ctx,inputs,output):
    metadata,program,source,target,operands,radial_map=inputs
    ctx.kernel_metadata,ctx.program=metadata,parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source,target,radial_map,*operands)


def compressed_backward(ctx,grad_outputs):
    source,target,radial_map,*operands=ctx.saved_tensors
    program,values,destinations=adjoint_program(ctx.program,operands,grad_outputs,
        gradient_mask(operands,ctx.needs_input_grad[4]),parse_metadata(ctx.kernel_metadata)[2])
    gradients=[None]*len(operands)
    if program:
        outputs=compressed_edge(ctx.kernel_metadata,repr(program),source,target,values,radial_map)
        for index,slot in destinations.items():gradients[index]=outputs[slot]
    return None,None,None,None,gradients,None


compressed_edge.register_autograd(compressed_backward,setup_context=compressed_context)
