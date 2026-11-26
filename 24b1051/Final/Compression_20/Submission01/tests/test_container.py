"""Integrity regressions, including the published v1 raw-hash convention.

Run with: python -m unittest discover -s tests -v
The real published artifact must also be audited; synthetic roundtrips alone
cannot establish compatibility with its untagged raw-byte hashes.
"""
import hashlib
import io
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from compression.encode import _write_zlib_block
from decompression.container import MAGIC, Container
from decompression.restore import _packed_producer, restore


RAW = b"\x80\x3f\x00\x40" * 64


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scratch = self.root / 'scratch'
        self.scratch.mkdir()

    def make(self, encoded=None, *, digest=None, raw_size=None, domain=None,
             header_domain=None, mutate=None):
        encoded = zlib.compress(RAW) if encoded is None else encoded
        block = dict(codec='zlib', role='raw', dtype='BF16', offset=12,
                     size=len(encoded), raw_size=len(RAW) if raw_size is None else raw_size,
                     sha256=digest or hashlib.sha256(RAW).hexdigest())
        if domain is not None:
            block['sha256_domain'] = domain
        tensors = [dict(name=f'tensor.{i}', original_shard='model.safetensors',
                        blocks=[block.copy()] if i == 0 else [])
                   for i in range(738)]
        header = dict(format='cs6013-czip-packed-v1', tensors=tensors)
        if header_domain is not None:
            header['sha256_domain'] = header_domain
        if mutate:
            mutate(header)
        payload = json.dumps(header).encode()
        path = self.root / 'model.czip'
        path.write_bytes(MAGIC + struct.pack('<I', 1) + encoded + payload
                         + struct.pack('<QQ', 12 + len(encoded), len(payload)) + MAGIC)
        container = Container(path)
        return container, container.header['tensors'][0]['blocks'][0]

    def rejected(self, container, block, message):
        with self.assertRaisesRegex(ValueError, message):
            container.extract(block, self.scratch)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_legacy_raw_hash(self):
        container, block = self.make()
        path = container.extract(block, self.scratch)
        self.assertEqual(path.read_bytes(), RAW)

    def test_legacy_compressed_hash(self):
        encoded = zlib.compress(RAW)
        container, block = self.make(encoded, digest=hashlib.sha256(encoded).hexdigest())
        self.assertEqual(container.extract(block, self.scratch).read_bytes(), RAW)

    def test_current_writer(self):
        handle = io.BytesIO(MAGIC + struct.pack('<I', 1))
        handle.seek(12)
        block = _write_zlib_block(handle, [RAW[:31], RAW[31:]], role='raw', dtype='BF16', shape=[128])
        self.assertEqual(block['sha256_domain'], 'compressed')
        encoded = handle.getvalue()[12:]
        self.assertEqual(block['sha256'], hashlib.sha256(encoded).hexdigest())
        container, block = self.make(encoded, digest=block['sha256'], domain=block['sha256_domain'])
        self.assertEqual(container.extract(block, self.scratch).read_bytes(), RAW)

    def test_explicit_raw_domain(self):
        container, block = self.make(domain='raw')
        self.assertEqual(container.extract(block, self.scratch).read_bytes(), RAW)

    def test_header_domain(self):
        container, block = self.make(header_domain='raw')
        self.assertEqual(container.extract(block, self.scratch).read_bytes(), RAW)

    def test_block_domain_overrides_header(self):
        container, block = self.make(header_domain='compressed', domain='raw')
        self.assertEqual(container.extract(block, self.scratch).read_bytes(), RAW)

    def test_explicit_domain_does_not_fall_back(self):
        container, block = self.make(domain='compressed')
        self.rejected(container, block, 'SHA256 mismatch')

    def test_unknown_domain(self):
        container, block = self.make(domain='guess')
        self.rejected(container, block, 'unsupported SHA256 domain')

    def test_hash_mismatch_reports_context_and_both_observations(self):
        container, block = self.make(digest='0' * 64)
        with self.assertRaises(ValueError) as caught:
            container.extract(block, self.scratch)
        message = str(caught.exception)
        for value in ('tensor.0', 'raw', 'offset 12', '0' * 64,
                      hashlib.sha256(RAW).hexdigest(), hashlib.sha256(zlib.compress(RAW)).hexdigest()):
            self.assertIn(value, message)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_valid_zlib_with_changed_payload(self):
        container, block = self.make(zlib.compress(b'X' * len(RAW)))
        self.rejected(container, block, 'SHA256 mismatch')

    def test_corrupt_deflate(self):
        encoded = bytearray(zlib.compress(RAW))
        encoded[-1] ^= 1
        container, block = self.make(bytes(encoded))
        self.rejected(container, block, 'corrupt DEFLATE')

    def test_truncated_stream_even_if_raw_length_and_hash_match(self):
        # Missing Adler32 can still emit every raw byte; eof must be checked.
        container, block = self.make(zlib.compress(RAW)[:-4])
        self.rejected(container, block, 'incomplete zlib stream')

    def test_trailing_bytes_even_if_compressed_hash_matches(self):
        for suffix in (b'junk', zlib.compress(b'another stream')):
            with self.subTest(suffix=suffix):
                encoded = zlib.compress(RAW) + suffix
                container, block = self.make(encoded, digest=hashlib.sha256(encoded).hexdigest())
                self.rejected(container, block, 'trailing compressed data')

    def test_wrong_raw_length(self):
        for size in (len(RAW) - 1, len(RAW) + 1, -1):
            with self.subTest(size=size):
                container, block = self.make(raw_size=size)
                self.rejected(container, block, 'length|raw size')

    def test_physical_truncation_after_open(self):
        container, block = self.make()
        with container.path.open('r+b') as handle:
            handle.truncate(13)
        self.rejected(container, block, 'truncated compressed block|corrupt DEFLATE')

    def test_invalid_bounds(self):
        for changes in ({'offset': 11}, {'size': -1}, {'size': 0}, {'size': 10**9}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'invalid block bounds'):
                self.make(mutate=lambda h: h['tensors'][0]['blocks'][0].update(changes))

    def test_overlapping_blocks(self):
        def overlap(h):
            h['tensors'][1]['blocks'] = [h['tensors'][0]['blocks'][0].copy()]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.make(mutate=overlap)

    def test_packed_second_extraction_failure_cleans_first(self):
        class FailingContainer:
            def extract(inner, block, temporary):
                if block['role'] == 'scales':
                    raise ValueError('bad scales')
                path = temporary / 'qweight.raw'
                path.write_bytes(b'\x00')
                return path
        entry = {'blocks': [{'role': 'qweight'}, {'role': 'scales'}]}
        with self.assertRaisesRegex(ValueError, 'bad scales'):
            list(_packed_producer(FailingContainer(), entry, self.scratch))
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_restore_failure_cleans_partial_shards_and_marker(self):
        def prepare(h):
            h['metadata_files'] = ['config.json']
            original = h['tensors'][0]['blocks'][0]
            for i, entry in enumerate(h['tensors']):
                entry.update(mode='raw', shape=[len(RAW) // 2], numel=len(RAW) // 2,
                             original_shard='first.safetensors' if i == 0 else 'second.safetensors')
                # Only the first two tensors are reached before the failure.
                entry['blocks'] = [original.copy()]
            # Reusing offsets would fail bounds validation, so move each block.
        encoded = zlib.compress(RAW)
        container, block = self.make()
        h = container.header
        prepare(h)
        chunks = []
        offset = 12
        for i, entry in enumerate(h['tensors']):
            b = entry['blocks'][0]
            b['offset'] = offset
            if i == 1:
                b['sha256'] = '0' * 64
            offset += len(encoded)
            chunks.append(encoded)
        payload = json.dumps(h).encode()
        container.path.write_bytes(MAGIC + struct.pack('<I', 1) + b''.join(chunks)
                                   + payload + struct.pack('<QQ', offset, len(payload)) + MAGIC)
        (self.root / 'config.json').write_text('{}')
        output = self.root / 'restored'
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            restore(self.root, output)
        self.assertFalse(list(output.glob('*.safetensors')))
        self.assertFalse(list(output.glob('*.partial')))
        self.assertFalse((output / 'model.safetensors.index.json').exists())
        self.assertFalse((output / 'DECOMPRESSION_COMPLETE.json').exists())
        self.assertFalse((output / '.decompress_tmp').exists())

    def test_missing_metadata_cleans_scratch(self):
        self.make(mutate=lambda h: h.update(metadata_files=['missing.json']))
        output = self.root / 'restored'
        with self.assertRaises(FileNotFoundError):
            restore(self.root, output)
        self.assertFalse((output / '.decompress_tmp').exists())
        self.assertFalse(list(output.glob('*.partial')))
        self.assertFalse((output / 'DECOMPRESSION_COMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main()
