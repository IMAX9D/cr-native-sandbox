"""Bounded synthetic CPU/GPU inference check; never starts a battle or training."""
import argparse,json,time
from pathlib import Path
import torch
from .catalog import CardVocabulary
from .config import ModelConfig
from .model import R0Policy
from .observation import ObservationBuilder
from .synthetic import rich_public_frame
from .native_graph import native_definitions


def run():
    torch.set_num_threads(2);torch.manual_seed(3901)
    vocabulary=CardVocabulary.from_native();results=[]
    for name,config in [('reference',ModelConfig.reference(allow_incomplete_capture=True)),('elite',ModelConfig(allow_incomplete_capture=True))]:
        builder=ObservationBuilder(vocabulary,config);model=R0Policy(vocabulary,config).eval()
        batch=builder.batch([rich_public_frame(builder,side=side) for side in (0,1)])
        with torch.no_grad():cpu=model.encode(batch)
        result=dict(profile=name,parameters=sum(p.numel() for p in model.parameters()),width=config.width,layers=config.layers,
            heads=config.heads,lstm_hidden=config.hidden,core_input=model.core_input[0].in_features,
            learned_spatial=config.learned_scatter_channels,spatial_channels=config.spatial_channels,spatial_input=model.spatial.input.in_channels,
            residual_blocks=config.spatial_blocks,group_slots=config.local_pool_slots,cpu_finite=bool(torch.isfinite(cpu.policy).all()),
            native_graph_sha256=native_definitions()[1],formal_training=False,synthetic=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            model=model.cuda();gpu_batch=batch.to('cuda')
            with torch.no_grad():
                gpu=model.encode(gpu_batch);actions=model.decode(gpu_batch,gpu,sample=False)
            torch.cuda.synchronize()
            result.update(gpu_device=torch.cuda.get_device_name(),gpu_finite=bool(torch.isfinite(gpu.policy).all()) and bool(torch.isfinite(actions.logp).all()),
                cpu_gpu_policy_max_abs=float((cpu.policy-gpu.policy.cpu()).abs().max()),
                cpu_gpu_close=bool(torch.allclose(cpu.policy,gpu.policy.cpu(),atol=5e-4,rtol=1e-4)),
                peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,2),batch_size=batch.batch_size)
            del gpu,gpu_batch,actions
        results.append(result);del model,cpu,batch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if not all(r['cpu_finite'] and r.get('gpu_finite',True) and r.get('cpu_gpu_close',True) for r in results):raise RuntimeError(results)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();results=run();text=json.dumps(results,ensure_ascii=False,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
