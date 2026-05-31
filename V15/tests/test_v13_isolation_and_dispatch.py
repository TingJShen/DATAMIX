from pathlib import Path


V13_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = V13_ROOT.parent


def test_v13_dispatch_is_local_to_v13_main():
    v13_main = (V13_ROOT / "dapo" / "main_dapo.py").read_text(encoding="utf-8")
    v11_main = (SNAPSHOT_ROOT / "dapo" / "main_dapo.py").read_text(encoding="utf-8")

    assert "from .dapo_ray_trainer_v8 import RayDAPOTrainer" in v13_main
    assert "prototype_dual_value" not in v13_main
    assert "sample_taylor_v13" in v13_main
    assert "RayDAPOTrainerV13" in v13_main
    assert "full_dataset_v12" not in v13_main
    assert "sample_taylor_v13" not in v11_main
    assert "RayDAPOTrainerV13" not in v11_main


def test_v13_smoke_script_uses_isolated_paths_and_method():
    script = (V13_ROOT / "dynamic_train_v13_a100_smoke_bsz4_20step.sh").read_text(encoding="utf-8")

    assert 'WORK_DIR="${BASE_DIR}/v13"' in script
    assert 'RUNTIME_DIR="${BASE_DIR}/runtime_v13"' in script
    assert 'export RAY_TMPDIR="${BASE_DIR}/r_v13"' in script
    assert "DYNAMIC_METHOD=${DYNAMIC_METHOD:-sample_taylor_v13}" in script
    assert "OUTPUT_ROOT=${OUTPUT_ROOT:-./output_v13}" in script


def test_v13_technical_report_is_archived():
    report = V13_ROOT / "docs" / "v13_technical_report.md"

    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Taylor Link" in text
    assert "Verification Checklist" in text


def test_v13_formal_and_merge_scripts_are_present():
    formal_script = V13_ROOT / "dynamic_train_v13_a100_formal.sh"
    merge_script = V13_ROOT / "tools" / "model_merge.sh"
    merge_impl = V13_ROOT / "tools" / "model_merge.py"

    assert formal_script.exists()
    assert merge_script.exists()
    assert merge_impl.exists()
    assert "sample_taylor_v13" in formal_script.read_text(encoding="utf-8")
    assert "python -m tools.model_merge merge" in merge_script.read_text(encoding="utf-8")


def test_v13_does_not_keep_v12_curriculum_modules():
    assert not (V13_ROOT / "dapo" / "dapo_ray_trainer_v12.py").exists()
    assert not (V13_ROOT / "dapo" / "curriculum_sampler.py").exists()
