import json
from itertools import islice
from pathlib import Path
# Note: Just a utility file to load confrag, mapping LightRAG to Confrag indices and parsing structured model output. 

def load_pilot(path, count):
    rows = []
    with open(path, encoding='utf-8') as f:
        for number, line in enumerate(islice(f, count), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}: invalid JSON at physical line {number}: {e}') from e
    if len(rows) != count:
        raise ValueError(f'Expected {count} records, found {len(rows)}')
    if len({r['id'] for r in rows}) != count:
        raise ValueError('Duplicate question IDs')
    return rows


def source_map(rows):
    # EXACTLY reproduces server_load(1).py's counter, including unannotated sites.
    result = {}
    for row in rows:
        for web in row['websites']:
            if web['content']:
                result[str(len(result))] = {
                    'question_id': row['id'], 'index': web.get('index'),
                    'url': web.get('website'),
                }
    return result

import json
import re


def extract_prediction_json(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty model response")

    text = text.strip()

    # First accept an ordinary JSON response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove standalone Markdown fence lines, including unclosed fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    ).strip()

    # The model may have added commentary before the JSON.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    # Parse one complete object; retain surrounding text only in raw output.
    obj, end = json.JSONDecoder().raw_decode(text, start)

    if not isinstance(obj, dict) or "answers" not in obj:
        raise ValueError("Expected a JSON object containing 'answers'")

    # Avoid silently selecting one of several predictions.
    trailing = text[end:].strip()
    if re.search(r'[\{\[]', trailing):
        # References contain square brackets, so only reject a second
        # independently parseable JSON object/array.
        for match in re.finditer(r'[\{\[]', trailing):
            try:
                extra, _ = json.JSONDecoder().raw_decode(
                    trailing, match.start()
                )
            except json.JSONDecodeError:
                continue
            if isinstance(extra, dict) and "answers" in extra:
                raise ValueError("Multiple prediction objects found")

    return obj
def parse_prediction(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    obj = extract_prediction_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get('answers'), list):
        raise ValueError('Expected {"answers": [...]}')
    if len(obj['answers']) > 32:
        raise ValueError('Too many clusters (maximum 32)')
    for a in obj['answers']:
        if not isinstance(a, dict) or not isinstance(a.get('answer'), str):
            raise ValueError('Each cluster needs a string answer')
        if not isinstance(a.get('index'), list) or any(type(i) is not int for i in a['index']):
            raise ValueError('index must be a list of integer ConfRAG webpage indices')
        if not isinstance(a.get('reason'), list) or any(not isinstance(r, str) for r in a['reason']):
            raise ValueError('reason must be a list of strings')
        if len(a['reason']) > 100:
            raise ValueError('Too many reasons')
    return obj


def retrieved_indices(payload, row, mapping):
    found, unknown, off_question = set(), [], []
    for ref in payload.get('references') or []:
        path = str(ref.get('file_path', ''))
        # Loader used numeric file_source; basename handles server path wrapping.
        source = mapping.get(path) or mapping.get(Path(path).name)
        if source is None:
            unknown.append(path)
        elif source['question_id'] != row['id']:
            off_question.append(path)
        elif source['index'] is not None:
            found.add(source['index'])
    return sorted(found), unknown, off_question
