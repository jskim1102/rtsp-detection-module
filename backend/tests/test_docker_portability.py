"""Docker clone portability and startup contracts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = tuple(
    path
    for path in (ROOT / "docker-compose.yml", ROOT / "docker-compose.harness.yml")
    if path.exists()
)


def test_shell_entrypoint_is_normalized_and_invoked_through_posix_shell() -> None:
    dockerfile = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY --chmod=0755 docker-entrypoint.sh /app/docker-entrypoint.sh" in dockerfile
    assert "replace(b'\\r\\n', b'\\n')" in dockerfile
    assert "replace(b'\\r', b'\\n')" in dockerfile
    assert "removeprefix(b'\\xef\\xbb\\xbf')" in dockerfile
    assert 'ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]' in dockerfile


def test_compose_rebuilds_local_app_images_and_waits_for_backend_health() -> None:
    for compose_file in COMPOSE_FILES:
        compose = compose_file.read_text(encoding="utf-8")

        assert compose.count("pull_policy: build") == 2, compose_file.name
        assert "http://127.0.0.1:8000/api/health" in compose, compose_file.name
        assert "condition: service_healthy" in compose, compose_file.name


def test_git_forces_lf_for_shell_scripts() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "* text=auto eol=lf" in attributes
    assert "*.sh text eol=lf" in attributes
