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
            {"graphrag", "gradio", "pandas", "pyarrow", "pypdf"},
        )
        for package in pins:
            self.assertTrue(
                any(dependency.startswith(package) for dependency in dependencies),
                f"Docker pin {package} is missing from pyproject.toml",
            )

    def test_dockerfile_caches_dependencies_before_copying_source(self) -> None:
        dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")

        requirements_copy = dockerfile.index("COPY requirements.docker.txt")
        dependency_install = dockerfile.index("pip install -r requirements.docker.txt")
        source_copy = dockerfile.index("COPY src ./src")
        application_install = dockerfile.index("pip install --no-deps .")
        self.assertLess(requirements_copy, dependency_install)
        self.assertLess(dependency_install, source_copy)
        self.assertLess(source_copy, application_install)
        self.assertIn("--mount=type=cache,target=/root/.cache/pip", dockerfile)


if __name__ == "__main__":
    unittest.main()
