"""Streaming transport used only to recover exhausted API slots.

The prompt, effort, token cap, temperature and per-read timeout are unchanged.
Full actual request arguments and all failed attempts are retained.
"""
from __future__ import annotations
import json,os,time,uuid
from pathlib import Path
import client
from tools.audit_cn_mem_results import EFFORTS

_client=None

def complete(model,prompt,*,temperature,max_tokens,n,log_dir):
    global _client
    from openai import OpenAI
    if _client is None:
        _client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],timeout=600,max_retries=0)
    assert n==1, 'Recovery is restricted to one failed slot at a time'
    kind=client.model_cfg(model)['kind']
    request={'model':model,'messages':[{'role':'user','content':prompt}],'n':1,
             'stream':True,'stream_options':{'include_usage':True}}
    if kind=='reasoning':request.update(max_tokens=max(max_tokens*32,16384),reasoning_effort=EFFORTS[model])
    else:request.update(max_tokens=max_tokens,temperature=temperature)
    started=time.monotonic();errors=[];failed_streams=[];accepted=None
    for attempt in range(5):
        content=[];reasoning=[];usage=None;finish=None;response_id=None;returned_model=None
        try:
            with _client.chat.completions.create(**request) as stream:
                for chunk in stream:
                    response_id=chunk.id or response_id;returned_model=chunk.model or returned_model
                    if chunk.usage is not None:usage=chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.delta.content:content.append(choice.delta.content)
                        thought=getattr(choice.delta,'reasoning_content',None) or getattr(choice.delta,'reasoning',None)
                        if thought:reasoning.append(thought)
                        if choice.finish_reason:finish=choice.finish_reason
            if finish is None or not response_id:
                raise RuntimeError('Incomplete recovery stream: missing finish or response ID')
            accepted={'content':''.join(content),'reasoning_content':''.join(reasoning),
                      'finish_reason':finish,'response_id':response_id,'model_returned':returned_model,'usage':usage or {},'usage_missing':usage is None}
            break
        except Exception as error:
            message=str(error);key=os.environ.get('OPENAI_API_KEY')
            if key:message=message.replace(key,'[REDACTED]')
            if finish is not None and response_id:
                # A completed answer remains scientific evidence even if the
                # trailing telemetry frame disconnects. Never resample it.
                accepted={'content':''.join(content),'reasoning_content':''.join(reasoning),
                          'finish_reason':finish,'response_id':response_id,'model_returned':returned_model,
                          'usage':usage or {},'usage_missing':usage is None,
                          'post_finish_transport_warning':{'type':type(error).__name__,'message':message[:2000]}}
                break
            errors.append({'attempt':attempt+1,'type':type(error).__name__,'message':message[:2000]})
            failed_streams.append({'response_id':response_id,'model_returned':returned_model,
                                  'partial_content':''.join(content),'partial_reasoning':''.join(reasoning),
                                  'finish_reason':finish,'usage':usage})
            if attempt<4:time.sleep(min(2**attempt,16))
    record={'request':request,'transport':'streaming-recovery-v1','read_timeout_seconds':600,
            'call_id':uuid.uuid4().hex,'sample_index':0,'transport_errors':errors,
            'failed_streams':failed_streams,'attempts':len(errors)+int(accepted is not None),
            'latency_sec':round(time.monotonic()-started,3)}
    text=''
    if accepted is not None:
        text=accepted.pop('content');record.update(accepted)
    else:record['finish_reason']='transport_error'
    raw={'backend':'custom-streaming-recovery','model_requested':model,'kind':kind,'prompt':prompt,
         'temperature':temperature,'max_tokens':max_tokens,'n':1,'completions':[text],
         'finish_reasons':[record['finish_reason']],'per_sample':[record],'billed_usd':0.0,
         'latency_sec':round(time.monotonic()-started,3)}
    client._write_log(log_dir,raw)
    return [text]
