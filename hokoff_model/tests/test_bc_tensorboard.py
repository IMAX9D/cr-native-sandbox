import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
from scripts.mirror_bc_metrics_to_tensorboard import mirror_available


class MirrorTests(unittest.TestCase):
    def test_partial_line_no_duplicates_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metrics.jsonl'
            writer = Mock()
            self.assertEqual(mirror_available(path, writer), 0)
            path.write_bytes(b'{"phase":"train","step":100,"loss":4.5}\n{"phase":"validation","step":100,"loss":4')
            offset = mirror_available(path, writer)
            writer.add_scalar.assert_called_once_with('train/loss', 4.5, 100)
            self.assertEqual(mirror_available(path, writer, offset), offset)
            self.assertEqual(writer.add_scalar.call_count, 1)
            with path.open('ab') as stream:
                stream.write(b'.8}\n')
            offset = mirror_available(path, writer, offset)
            writer.add_scalar.assert_called_with('val/loss', 4.8, 100)
            self.assertEqual(offset, path.stat().st_size)
            path.write_bytes(b'')
            with self.assertRaisesRegex(RuntimeError, 'truncated'):
                mirror_available(path, writer, offset)


if __name__ == '__main__':
    unittest.main()
