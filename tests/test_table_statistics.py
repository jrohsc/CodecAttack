"""Check missing-data behavior and numeric reporting in changed table builders."""
import ast
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path,name', [
    ('scripts/build_cross_model_table.py','cell'),
    ('scripts/build_s1_codec_latex.py','_fmt_cell'),
    ('scripts/build_s1_latent_vs_waveform_table.py','fmt_cell'),
    ('scripts/build_s1_non_robust_table.py','_fmt_cell'),
    ('scripts/campaigns/build_watermark_table.py','cell'),
])
def test_single_run_tables_report_point_estimates_and_preserve_missing(path,name):
    tree=ast.parse((ROOT/path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),namespace)
    cell=namespace[name]
    assert cell(3,4)=='$75.0$'
    assert cell(0,4)=='$0.0$'
    assert cell(0,0)==r'\pending'
