import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
control=json.loads((ROOT/'cpu-gpu-control/summary.json').read_text())
profile=json.loads((ROOT/'cpu-gpu-profile/summary.json').read_text())
trace=json.loads((ROOT/'cpu-gpu-profile/trace-analysis.json').read_text())
gold_ref=json.loads((ROOT/'slots4equal16-summary.json').read_text())[0]
gold=json.loads((Path(gold_ref['source'])/'policy-0.json').read_text())['rows'][0]
assert control['finals']==profile['finals']==gold['finals'],'profiling changed a game result/action sequence'
wall=control['wall_seconds'];actor_cpu=control['actor_process_cpu_seconds']
wall_rows=[];cpu_rows=[]
for name,value in control['exclusive_phases'].items():
    wall_rows.append(dict(phase=name,seconds=value['wall_ns']/1e9,percent=value['wall_ns']/1e9/wall*100))
    cpu_rows.append(dict(phase=name,cpu_seconds=value['cpu_ns']/1e9,percent=value['cpu_ns']/1e9/actor_cpu*100))
wall_rows.append(dict(phase='loop_and_measurement_other',seconds=wall-sum(x['seconds'] for x in wall_rows),percent=0))
wall_rows[-1]['percent']=wall_rows[-1]['seconds']/wall*100
n=profile['native_delta']['native']['metrics'];j=profile['native_delta']['java']
cpu=lambda key:n[key]['cpu_ns']/1e9
java_cpu=lambda key:j[key]['cpu_ns']/1e9
server_total=java_cpu('request_parse')+java_cpu('handler_total')+java_cpu('response_serialize')+java_cpu('socket_write')
server={
    'engine_update':cpu('engine_update'),
    'terminal_capture':cpu('capture_step')+cpu('capture_observe'),
    'observation_read_and_native_json':cpu('observe_total')-cpu('capture_observe')-cpu('episode_json_observe'),
    'episode_json':cpu('episode_json_step')+cpu('episode_json_observe'),
    'binding_and_handle_setup':cpu('bind_unbind'),
    'step_wrapper_other':cpu('step_total')-cpu('engine_update')-cpu('capture_step')-cpu('episode_json_step'),
    'action_execution':cpu('action'),'legality_grid':cpu('legality_grid'),
    'java_and_jni_wrapper_other':java_cpu('handler_total')-sum(cpu(k) for k in ('step_total','observe_total','action','legality_grid','bind_unbind','create')),
    'request_parse':java_cpu('request_parse'),'response_serialize':java_cpu('response_serialize'),'socket_write':java_cpu('socket_write')}
assert abs(sum(server.values())-server_total)<1e-6
result=dict(control_wall=wall,control_actor_cpu=actor_cpu,wall_breakdown=sorted(wall_rows,key=lambda x:-x['seconds']),
            actor_cpu_breakdown=sorted(cpu_rows,key=lambda x:-x['cpu_seconds']),
            server_profile_cpu=server_total,server_cpu_breakdown=[dict(phase=k,cpu_seconds=v,percent=v/server_total*100) for k,v in sorted(server.items(),key=lambda x:-x[1])],
            profiled_wall=profile['wall_seconds'],trace_export_wall=profile['profiler_step_wall_seconds'],
            remaining_profile_overhead_ratio=(profile['wall_seconds']-profile['profiler_step_wall_seconds'])/wall-1,
            all_actions_and_results_equal=True,trace=trace,
            caveat='Actor wall/CPU tables use no-CUDA-trace control. Native CPU table comes from instrumented run. GPU activity is sampled; do not sum these into one hardware percentage.')
(ROOT/'full-profile-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='trace'}))
