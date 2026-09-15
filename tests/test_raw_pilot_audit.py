import io
import json
from pathlib import Path
import tempfile
import unittest
import zstandard as zstd
from data_pipeline.audit_raw_pilot import audit
from data_pipeline.prepare_600k import encoded


def frame(tick,events=(),present=True):
    return dict(kind='frame',state=dict(tick=tick,entities=[dict(generation_key=1,id='0x1',kind=14,side=0,hp=100,max_hp=100)] if present else [],projectiles=[],effects=[]),
                details=dict(objects=[dict(key=1,native_name='Knight',source_key=0)] if present else []),
                telemetry=dict(epoch=1,events=list(events)),capture_quality_flags=[])


class RawAuditTests(unittest.TestCase):
    def test_fact_preservation_and_causal_damage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'raw.zst'
            events=[dict(sequence=1,tick=1,effective_damage=123456789,attacker_key=1,target_key=1,target_is_tower=0,lethal=1),
                    dict(sequence=2,tick=50,effective_damage=777,attacker_key=1,target_key=1,target_is_tower=0,lethal=0),
                    dict(sequence=3,tick=1,effective_damage=9,attacker_key=0,target_key=1,target_is_tower=0,lethal=0)]
            with src.open('wb') as f,zstd.ZstdCompressor().stream_writer(f) as writer:
                for r in [frame(0),frame(1,events),frame(5),frame(6,present=False)]:writer.write(encoded(r))
            report=audit(src,root/'audit')
            self.assertEqual(report['issues']['future_damage_event'],1)
            self.assertEqual(report['issues']['damage_attacker_unknown'],1)
            with (root/'audit'/'lifecycle.jsonl.zst').open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as raw,io.TextIOWrapper(raw) as text:
                record=json.loads(next(text))
            self.assertFalse(record['death_confirmed']);self.assertTrue(record['lethal_damage_observed'])
            with (root/'audit'/'damage-history-5tick.jsonl.zst').open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as raw,io.TextIOWrapper(raw) as text:
                rows=[json.loads(line) for line in text]
            self.assertEqual(rows[0]['rows'][0]['values'][0],0)
            self.assertEqual(rows[1]['rows'][0]['values'][:3],[123456789]*3)


if __name__=='__main__':unittest.main()
