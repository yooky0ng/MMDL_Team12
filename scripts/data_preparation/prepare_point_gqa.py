"""Build paired 4K SFT inputs and an image-disjoint GQA validation set."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import io
import json
from pathlib import Path
import random
import re
import struct
import zipfile
import zlib
from threading import local

import ijson
import requests

GQA_URL = 'https://downloads.cs.stanford.edu/nlp/data/gqa'


class RemoteZip(io.RawIOBase):
    """Seekable HTTP range reader for locating files in an official ZIP."""

    def __init__(self, name):
        self.url = f'{GQA_URL}/{name}'
        response = requests.head(self.url, timeout=20)
        response.raise_for_status()
        self.size = int(response.headers['Content-Length'])
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def fetch(self, start, size):
        end = start + size - 1
        for attempt in range(4):
            try:
                with requests.get(
                    self.url, params={'range': f'{start}-{end}'},
                    headers={'Range': f'bytes={start}-{end}'}, timeout=(10, 120),
                    stream=True,
                ) as response:
                    if (response.status_code != 206 or response.headers.get('Content-Range')
                            != f'bytes {start}-{end}/{self.size}'):
                        raise IOError(f'Invalid range response ({response.status_code}): {self.url}')
                    payload = response.content
                    if len(payload) != size:
                        raise IOError(f'Incomplete range for {self.url}')
                    return payload
            except (requests.RequestException, OSError):
                if attempt == 3:
                    raise

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b''
        payload = self.fetch(self.position, size)
        self.position += size
        return payload

    def member(self, info):
        header_size = 30
        # Fetch the fixed header, filename, bounded extra field and payload together.
        start = info.header_offset
        requested = header_size + len(info.filename.encode()) + 256 + info.compress_size
        data = self.fetch(start, min(requested, self.size - start))
        name_size, extra_size = struct.unpack_from('<HH', data, 26)
        payload_start = header_size + name_size + extra_size
        stop = payload_start + info.compress_size
        if stop > len(data):
            data += self.fetch(start + len(data), stop - len(data))
        raw = data[payload_start:stop]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(raw, -15)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f'Unsupported ZIP compression for {info.filename}')
        if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
            raise ValueError(f'Invalid ZIP member: {info.filename}')
        return raw


@contextmanager
def open_json_member(archive, local, basename):
    if local.is_file() and zipfile.is_zipfile(local):
        with zipfile.ZipFile(local) as zf:
            name = next(n for n in zf.namelist() if Path(n).name == basename)
            with zf.open(name) as handle:
                yield handle
        return
    reader = RemoteZip(archive)
    with zipfile.ZipFile(reader) as zf:
        info = next(i for i in zf.infolist() if Path(i.filename).name == basename)
    yield io.BytesIO(reader.member(info))


def candidate_questions(questions, graphs, seed, limit):
    rng = random.Random(seed)
    pool = []
    for qid, q in ijson.kvitems(questions, '', use_float=True):
        ops = q.get('semantic', [])
        if not ops or ops[-1].get('operation') != 'query':
            continue
        attribute = ops[-1].get('argument')
        if attribute not in {'color', 'material', 'shape', 'size'}:
            continue
        answer = q['answer'].lower()
        scene = graphs.get(q['imageId'])
        candidates = re.findall(r'\((\d+)\)', ops[-2].get('argument', '')) if len(ops) > 1 else []
        if scene is None or len(candidates) != 1:
            continue
        obj_id = candidates[0]
        obj = scene['objects'].get(obj_id)
        if (not obj or answer not in obj['attributes']
                or any(value is None for value in (obj['x'], obj['y'], obj['w'], obj['h']))
                or min(scene['width'], scene['height'], obj['w'], obj['h']) <= 0):
            continue
        box = [obj[key] for key in ('x', 'y', 'w', 'h')]
        width, height = scene['width'], scene['height']
        x, y, w, h = box
        if min(x, y) < 0 or x + w > width or y + h > height:
            continue
        if min(w / width, h / height) < .025:
            continue
        point = [round(100 * (x + w / 2) / width, 1), round(100 * (y + h / 2) / height, 1)]
        pool.append({'id': f'gqa_{qid}', 'image_id': q['imageId'], 'question': q['question'],
                     'answer': answer, 'object': obj['name'], 'attribute': attribute,
                     'box': box, 'point': point, 'image_size': [width, height]})
    rng.shuffle(pool)
    selected = {}
    for item in pool:
        selected.setdefault(item['image_id'], item)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f'Only {len(selected)} eligible images, expected {limit}')
    print(f'{limit} selected from {len(pool)} eligible questions', flush=True)
    return list(selected.values())


def row(item, image_root, grounded):
    answer = item['answer']
    target = f'<answer>{answer}</answer>'
    if grounded:
        target = (f"<think>\nThe {item['object']} is here. "
                  f"[point: {json.dumps([item['point']])}] "
                  f"Its {item['attribute']} is {answer}.\n</think>\n{target}")
    return {'id': item['id'], 'source': 'gqa', 'source_dataset': 'gqa_train_balanced',
            'qwen3vl': {'image': str(image_root / f"{item['image_id']}.jpg"),
                        'conversations': [{'from': 'human', 'value': '<image>\n' + item['question']},
                                          {'from': 'gpt', 'value': target}]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--point-data', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selection = root / 'gqa_selection.json'
    if selection.exists():
        selected = json.loads(selection.read_text())
    else:
        with open_json_member('sceneGraphs.zip', root / 'downloads/sceneGraphs.zip',
                              'train_sceneGraphs.json') as handle:
            graphs = json.load(handle)
        print('Loaded GQA training scene graphs', flush=True)
        with open_json_member('questions1.2.zip', root / 'downloads/questions1.2.zip',
                              'train_balanced_questions.json') as handle:
            selected = candidate_questions(handle, graphs, seed=42, limit=2300)
        selection.write_text(json.dumps(selected, indent=2))
    assert len(selected) == len({r['image_id'] for r in selected}) == 2300

    image_root = root / 'images'
    image_root.mkdir(exist_ok=True)
    pending = [i for i in selected if not (image_root / f"{i['image_id']}.jpg").is_file()]
    if pending:
        reader = RemoteZip('images.zip')
        with zipfile.ZipFile(reader) as zf:
            archive = {Path(info.filename).stem: info for info in zf.infolist()
                       if info.filename.endswith('.jpg')}

        worker = local()

        def download(item):
            ident = item['image_id']
            target = image_root / f'{ident}.jpg'
            temp = target.with_suffix('.partial')
            info = archive[ident]
            if not hasattr(worker, 'session'):
                worker.session = requests.Session()
            payload = None
            # Visual Genome hosts the same originals. Verify bytes against the GQA ZIP.
            for directory in ('VG_100K', 'VG_100K_2'):
                try:
                    url = f'https://cs.stanford.edu/people/rak248/{directory}/{ident}.jpg'
                    with worker.session.get(url, timeout=(5, 12)) as response:
                        if response.ok and zlib.crc32(response.content) == info.CRC:
                            payload = response.content
                            break
                except requests.RequestException:
                    continue
            if payload is None:
                payload = reader.member(info)
            temp.write_bytes(payload)
            temp.replace(target)

        with ThreadPoolExecutor(max_workers=12) as pool:
            for count, _ in enumerate(pool.map(download, pending), 1):
                if count % 100 == 0 or count == len(pending):
                    print(f'Images downloaded {count}/{len(pending)}', flush=True)

    point = [r for r in json.loads(args.point_data.read_text()) if r['source'] == 'point_cot']
    common = random.Random(42).sample(point, 2000)
    for name, grounded in [('plain', False), ('grounded', True)]:
        destination = root / name
        destination.mkdir(exist_ok=True)
        data = common + [row(item, image_root, grounded) for item in selected[:2000]]
        (destination / 'train.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (root / 'validation.json').write_text(json.dumps(selected[2000:], indent=2))
    print('Prepared two paired 4000-item training sets and 300 held-out GQA images', flush=True)


if __name__ == '__main__':
    main()
