import tomllib
import unittest
from pathlib import Path


class PackagingConfigurationTests(unittest.TestCase):
    def test_docker_pins_match_supported_dependency_families(self) -> None:
        root = Path(__file__).parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        dependencies = project["dependencies"]
        docker_requirements = (root / "requirements.docker.txt").read_text(encoding="utf-8").splitlines()
        pins = {line.split("==", 1)[0]: line for line in docker_requirements if line and not line.startswith("#")}

        self.assertEqual(
            set(pins),
            {"graphrag", "gradio", "pandas", "pyarrow", "pymupdf"},
        )
        for package in pins:
            self.assertTrue(
                any(dependency.startswith(package) for dependency in dependencies),
                f"Docker pin {package} is missing from pyproject.toml",
            )

    def test_dockerfile_caches_dependencies_before_copying_source(self) -> None:
        dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")

        requirements_copy = dockerfile.index("COPY requirements.docker.txt")
        dependency_install = dockerfile.index("uv pip install --system")
        source_copy = dockerfile.index("COPY . .")
        application_install = dockerfile.index("uv pip install --system --no-deps .")
        self.assertLess(requirements_copy, dependency_install)
        self.assertLess(dependency_install, source_copy)
        self.assertLess(source_copy, application_install)
        self.assertIn(
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/",
            dockerfile,
        )
        self.assertNotIn("pip install uv", dockerfile)
        self.assertIn('--default-index "${PYPI_INDEX_URL}"', dockerfile)
        self.assertGreaterEqual(
            dockerfile.count("--mount=type=cache,target=/root/.cache/uv"),
            2,
        )
        self.assertIn("# syntax=docker/dockerfile:1.7", dockerfile)
        self.assertIn(
            'ARG PYPI_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"',
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
