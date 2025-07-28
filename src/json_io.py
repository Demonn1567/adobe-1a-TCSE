import json, pathlib, orjson
from jsonschema import validate


_SCHEMAPATH = pathlib.Path(__file__).parent.parent / "sample_dataset/schema/output_schema.json"
_SCHEMA = json.loads(_SCHEMAPATH.read_text())


def write_json(data: dict, out_path: pathlib.Path):
    validate(instance=data, schema=_SCHEMA)      
    out_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
