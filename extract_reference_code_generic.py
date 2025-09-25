#!/usr/bin/env python3
"""
Generic Reference Code Extractor

Automatically discovers project dependencies and extracts relevant code from
reference repositories. Works with any project structure.

Usage:
    python extract_reference_code_generic.py [--project-dir PATH] [--references-dir PATH] [--output-dir PATH]

Options:
    --project-dir     : Root directory of your project (default: current directory)
    --references-dir  : Directory containing reference git repositories (default: ./reference_repos)
    --output-dir      : Output directory for extracted code (default: ./extracted_reference_code)
    --config-file     : Custom configuration file (default: auto-generate)
    --update-refs     : Update git repositories first
    --clean           : Remove existing output directory
"""

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import toml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GenericDependencyAnalyzer:
    """Analyze project dependencies from various file formats"""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.frontend_deps: Set[str] = set()
        self.backend_deps: Set[str] = set()
        self.all_deps: Set[str] = set()

    def analyze_all_dependencies(self) -> Set[str]:
        """Analyze all dependency files in the project"""

        # JavaScript/Node.js projects
        self.analyze_package_json()
        self.analyze_yarn_lock()
        self.analyze_pnpm_lock()

        # Python projects
        self.analyze_pyproject_toml()
        self.analyze_requirements_txt()
        self.analyze_poetry_lock()
        self.analyze_pipfile()
        self.analyze_setup_py()

        # Other languages
        self.analyze_cargo_toml()  # Rust
        self.analyze_go_mod()  # Go
        self.analyze_composer_json()  # PHP
        self.analyze_gemfile()  # Ruby
        self.analyze_build_gradle()  # Java/Kotlin
        self.analyze_csproj()  # C#

        self.all_deps = self.frontend_deps | self.backend_deps
        logger.info(f"Found {len(self.all_deps)} total dependencies across all files")
        return self.all_deps

    def analyze_package_json(self):
        """Analyze package.json files (can be multiple in monorepos)"""
        for package_json in self.project_dir.rglob("package.json"):
            if "node_modules" in str(package_json):
                continue

            try:
                with open(package_json) as f:
                    data = json.load(f)

                deps = set()
                for dep_type in [
                    "dependencies",
                    "devDependencies",
                    "peerDependencies",
                    "optionalDependencies",
                ]:
                    if dep_type in data:
                        deps.update(data[dep_type].keys())

                self.frontend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in {package_json}")

            except Exception as e:
                logger.warning(f"Could not parse {package_json}: {e}")

    def analyze_pyproject_toml(self):
        """Analyze pyproject.toml files"""
        for pyproject_file in self.project_dir.rglob("pyproject.toml"):
            try:
                with open(pyproject_file) as f:
                    data = toml.load(f)

                deps = set()

                # Standard dependencies
                if "project" in data and "dependencies" in data["project"]:
                    for dep in data["project"]["dependencies"]:
                        package_name = self._extract_package_name(dep)
                        if package_name:
                            deps.add(package_name)

                # Poetry dependencies
                if (
                    "tool" in data
                    and "poetry" in data["tool"]
                    and "dependencies" in data["tool"]["poetry"]
                ):
                    deps.update(data["tool"]["poetry"]["dependencies"].keys())

                # Development dependencies
                if "dependency-groups" in data:
                    for group, group_deps in data["dependency-groups"].items():
                        for dep in group_deps:
                            package_name = self._extract_package_name(dep)
                            if package_name:
                                deps.add(package_name)

                # Remove Python itself
                deps.discard("python")

                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in {pyproject_file}")

            except Exception as e:
                logger.warning(f"Could not parse {pyproject_file}: {e}")

    def analyze_requirements_txt(self):
        """Analyze requirements.txt files"""
        for req_file in self.project_dir.rglob("requirements*.txt"):
            try:
                with open(req_file) as f:
                    lines = f.readlines()

                deps = set()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("-"):
                        package_name = self._extract_package_name(line)
                        if package_name:
                            deps.add(package_name)

                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in {req_file}")

            except Exception as e:
                logger.warning(f"Could not parse {req_file}: {e}")

    def analyze_yarn_lock(self):
        """Extract package names from yarn.lock"""
        yarn_lock = self.project_dir / "yarn.lock"
        if yarn_lock.exists():
            try:
                with open(yarn_lock) as f:
                    content = f.read()

                # Extract package names from yarn.lock format
                package_pattern = r'^"?([^@\s"]+)(?:@[^"]*)?[^"]*"?:'
                matches = re.findall(package_pattern, content, re.MULTILINE)

                deps = set(matches)
                self.frontend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in yarn.lock")

            except Exception as e:
                logger.warning(f"Could not parse yarn.lock: {e}")

    def analyze_pnpm_lock(self):
        """Extract package names from pnpm-lock.yaml"""
        pnpm_lock = self.project_dir / "pnpm-lock.yaml"
        if pnpm_lock.exists():
            try:
                # Simple regex approach for YAML
                with open(pnpm_lock) as f:
                    content = f.read()

                # Look for package entries
                package_pattern = r"^\s+([^:\s/]+):"
                matches = re.findall(package_pattern, content, re.MULTILINE)

                deps = set(matches)
                self.frontend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in pnpm-lock.yaml")

            except Exception as e:
                logger.warning(f"Could not parse pnpm-lock.yaml: {e}")

    def analyze_poetry_lock(self):
        """Extract package names from poetry.lock"""
        poetry_lock = self.project_dir / "poetry.lock"
        if poetry_lock.exists():
            try:
                with open(poetry_lock) as f:
                    content = f.read()

                # Extract package names from poetry.lock format
                package_pattern = r'^\[\[package\]\]\nname = "([^"]+)"'
                matches = re.findall(package_pattern, content, re.MULTILINE)

                deps = set(matches)
                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in poetry.lock")

            except Exception as e:
                logger.warning(f"Could not parse poetry.lock: {e}")

    def analyze_pipfile(self):
        """Analyze Pipfile"""
        pipfile = self.project_dir / "Pipfile"
        if pipfile.exists():
            try:
                with open(pipfile) as f:
                    data = toml.load(f)

                deps = set()
                for section in ["packages", "dev-packages"]:
                    if section in data:
                        deps.update(data[section].keys())

                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in Pipfile")

            except Exception as e:
                logger.warning(f"Could not parse Pipfile: {e}")

    def analyze_setup_py(self):
        """Extract dependencies from setup.py"""
        setup_py = self.project_dir / "setup.py"
        if setup_py.exists():
            try:
                with open(setup_py) as f:
                    content = f.read()

                # Look for install_requires and extras_require
                install_pattern = r"install_requires\s*=\s*\[(.*?)\]"
                matches = re.findall(install_pattern, content, re.DOTALL)

                deps = set()
                for match in matches:
                    # Extract package names from strings
                    package_strings = re.findall(r'["\']([^"\']+)["\']', match)
                    for pkg_str in package_strings:
                        package_name = self._extract_package_name(pkg_str)
                        if package_name:
                            deps.add(package_name)

                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} deps in setup.py")

            except Exception as e:
                logger.warning(f"Could not parse setup.py: {e}")

    def analyze_cargo_toml(self):
        """Analyze Cargo.toml for Rust projects"""
        cargo_toml = self.project_dir / "Cargo.toml"
        if cargo_toml.exists():
            try:
                with open(cargo_toml) as f:
                    data = toml.load(f)

                deps = set()
                for section in [
                    "dependencies",
                    "dev-dependencies",
                    "build-dependencies",
                ]:
                    if section in data:
                        deps.update(data[section].keys())

                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} Rust deps in Cargo.toml")

            except Exception as e:
                logger.warning(f"Could not parse Cargo.toml: {e}")

    def analyze_go_mod(self):
        """Analyze go.mod for Go projects"""
        go_mod = self.project_dir / "go.mod"
        if go_mod.exists():
            try:
                with open(go_mod) as f:
                    content = f.read()

                # Extract require statements
                require_pattern = r"^\s+([^\s]+)\s+v[\d\.]+"
                matches = re.findall(require_pattern, content, re.MULTILINE)

                deps = set(matches)
                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} Go deps in go.mod")

            except Exception as e:
                logger.warning(f"Could not parse go.mod: {e}")

    def analyze_composer_json(self):
        """Analyze composer.json for PHP projects"""
        composer_json = self.project_dir / "composer.json"
        if composer_json.exists():
            try:
                with open(composer_json) as f:
                    data = json.load(f)

                deps = set()
                for section in ["require", "require-dev"]:
                    if section in data:
                        deps.update(data[section].keys())

                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} PHP deps in composer.json")

            except Exception as e:
                logger.warning(f"Could not parse composer.json: {e}")

    def analyze_gemfile(self):
        """Analyze Gemfile for Ruby projects"""
        gemfile = self.project_dir / "Gemfile"
        if gemfile.exists():
            try:
                with open(gemfile) as f:
                    content = f.read()

                # Extract gem statements
                gem_pattern = r"gem\s+['\"]([^'\"]+)['\"]"
                matches = re.findall(gem_pattern, content)

                deps = set(matches)
                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} Ruby gems in Gemfile")

            except Exception as e:
                logger.warning(f"Could not parse Gemfile: {e}")

    def analyze_build_gradle(self):
        """Analyze build.gradle for Java/Kotlin projects"""
        for gradle_file in self.project_dir.rglob("build.gradle*"):
            try:
                with open(gradle_file) as f:
                    content = f.read()

                # Extract dependency declarations
                dep_pattern = r'(?:implementation|api|compile|testImplementation)\s+["\']([^:"\']+):([^:"\']+)'
                matches = re.findall(dep_pattern, content)

                deps = set()
                for group, artifact in matches:
                    deps.add(f"{group}:{artifact}")

                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} Java/Kotlin deps in {gradle_file}")

            except Exception as e:
                logger.warning(f"Could not parse {gradle_file}: {e}")

    def analyze_csproj(self):
        """Analyze .csproj files for C# projects"""
        for csproj_file in self.project_dir.rglob("*.csproj"):
            try:
                with open(csproj_file) as f:
                    content = f.read()

                # Extract PackageReference items
                package_pattern = r'<PackageReference\s+Include="([^"]+)"'
                matches = re.findall(package_pattern, content)

                deps = set(matches)
                self.backend_deps.update(deps)
                logger.info(f"Found {len(deps)} C# packages in {csproj_file}")

            except Exception as e:
                logger.warning(f"Could not parse {csproj_file}: {e}")

    def _extract_package_name(self, dep_string: str) -> Optional[str]:
        """Extract clean package name from dependency string"""
        if not dep_string:
            return None

        # Remove version specifiers and extras
        # Examples: "package>=1.0.0" -> "package", "package[extra]" -> "package"
        clean_name = re.split(r"[>=<!\[\s]", dep_string.strip())[0]

        # Remove quotes
        clean_name = clean_name.strip("'\"")

        # Skip if empty or looks like a version/constraint
        if not clean_name or re.match(r"^[\d\.]", clean_name):
            return None

        return clean_name


class GenericReferenceDiscovery:
    """Discover reference repositories for project dependencies"""

    def __init__(self, references_dir: Path, dependencies: Set[str]):
        self.references_dir = references_dir
        self.dependencies = dependencies
        self.mappings: Dict[str, Path] = {}

    def discover_references(self) -> Dict[str, Path]:
        """Discover which reference repositories match project dependencies"""

        if not self.references_dir.exists():
            logger.warning(f"References directory {self.references_dir} does not exist")
            return {}

        # Direct name matching
        self._find_direct_matches()

        # Pattern matching
        self._find_pattern_matches()

        # Subdirectory matching (for monorepos)
        self._find_subdirectory_matches()

        logger.info(
            f"Mapped {len(self.mappings)} dependencies to reference repositories"
        )
        return self.mappings

    def _find_direct_matches(self):
        """Find exact name matches"""
        for dep in self.dependencies:
            dep_path = self.references_dir / dep
            if dep_path.exists() and dep_path.is_dir():
                self.mappings[dep] = dep_path
                logger.debug(f"Direct match: {dep} -> {dep_path}")

    def _find_pattern_matches(self):
        """Find matches using pattern matching"""
        for dep in self.dependencies:
            if dep in self.mappings:
                continue

            # Try various patterns
            patterns = [
                dep.replace("-", "_"),  # package-name -> package_name
                dep.replace("_", "-"),  # package_name -> package-name
                dep.replace("@", ""),  # @scope/package -> scopepackage
                dep.split("/")[-1],  # @scope/package -> package
                dep.lower(),
                dep.replace(".", "-"),  # some.package -> some-package
            ]

            for pattern in patterns:
                pattern_path = self.references_dir / pattern
                if pattern_path.exists() and pattern_path.is_dir():
                    self.mappings[dep] = pattern_path
                    logger.debug(f"Pattern match: {dep} -> {pattern_path}")
                    break

    def _find_subdirectory_matches(self):
        """Find matches in subdirectories (for organized reference repos)"""
        for dep in self.dependencies:
            if dep in self.mappings:
                continue

            # Search in common subdirectory structures
            search_paths = [
                self.references_dir / "python" / dep,
                self.references_dir / "javascript" / dep,
                self.references_dir / "node_modules" / dep,
                self.references_dir / "packages" / dep,
                self.references_dir / "integrations" / dep,
            ]

            for search_path in search_paths:
                if search_path.exists() and search_path.is_dir():
                    self.mappings[dep] = search_path
                    logger.debug(f"Subdirectory match: {dep} -> {search_path}")
                    break

            # Search recursively for partial matches
            if dep not in self.mappings:
                self._search_recursive_matches(dep)

    def _search_recursive_matches(self, dep: str):
        """Search recursively for repositories that might match"""
        dep_lower = dep.lower().replace("-", "").replace("_", "")

        for item in self.references_dir.rglob("*"):
            if not item.is_dir():
                continue

            item_name = item.name.lower().replace("-", "").replace("_", "")

            # Check if dependency name is contained in directory name
            if dep_lower in item_name or item_name in dep_lower:
                # Avoid deep nested paths and common directories
                if len(item.parts) - len(self.references_dir.parts) <= 3:
                    if item.name not in [
                        "src",
                        "lib",
                        "dist",
                        "build",
                        "node_modules",
                        ".git",
                    ]:
                        self.mappings[dep] = item
                        logger.debug(f"Recursive match: {dep} -> {item}")
                        break


class GenericCodeExtractor:
    """Extract code from reference repositories"""

    # Universal patterns
    ALWAYS_INCLUDE = {
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "HISTORY.md",
        "pyproject.toml",
        "package.json",
        "composer.json",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "*.csproj",
        "src/",
        "lib/",
        "index.ts",
        "index.js",
        "__init__.py",
        "main.py",
    }

    EXCLUDE_DIRS = {
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
        ".coverage",
        "htmlcov",
        "docs/_build",
        "site-packages",
        "tests",
        "test",
        "examples",
        "example",
        "docs",
        "doc",
        "benchmarks",
        "benchmark",
        ".github",
        ".vscode",
        "coverage",
        "target",
        "bin",
        "obj",
        ".next",
        ".nuxt",
        "vendor",
    }

    EXCLUDE_PATTERNS = {
        "*.pyc",
        "*.pyo",
        "*.egg-info",
        "*.log",
        "*.tmp",
        "*.cache",
        ".DS_Store",
        "Thumbs.db",
        "*.swp",
        "*.swo",
        "*~",
        "*.test.js",
        "*.test.ts",
        "*.spec.js",
        "*.spec.ts",
        "test_*.py",
        "*_test.py",
        "*.test.go",
        "*_test.go",
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def extract_all(self, mappings: Dict[str, Path]):
        """Extract code from all mapped repositories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        extracted_count = 0
        total_files = 0

        for dep_name, repo_path in mappings.items():
            logger.info(f"Extracting {dep_name} from {repo_path}")
            dest_path = self.output_dir / dep_name
            file_count = self._extract_repository(repo_path, dest_path, dep_name)

            if file_count > 0:
                extracted_count += 1
                total_files += file_count

        logger.info(
            f"Extracted {extracted_count} repositories with {total_files} total files"
        )
        return extracted_count, total_files

    def _extract_repository(
        self, source_dir: Path, dest_dir: Path, lib_name: str
    ) -> int:
        """Extract core files from a single repository"""
        dest_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = 0

        # Copy essential root files first
        for item in source_dir.iterdir():
            if item.is_file() and self._should_include_file(item, is_root=True):
                try:
                    shutil.copy2(item, dest_dir / item.name)
                    extracted_files += 1
                except Exception as e:
                    logger.warning(f"Could not copy {item}: {e}")

        # Then copy source directories and files
        for item in source_dir.rglob("*"):
            # Skip if parent directory should be excluded
            if any(exclude_dir in item.parts for exclude_dir in self.EXCLUDE_DIRS):
                continue

            # Skip if matches exclude pattern
            if any(item.match(pattern) for pattern in self.EXCLUDE_PATTERNS):
                continue

            # Calculate relative path from source
            rel_path = item.relative_to(source_dir)
            dest_path = dest_dir / rel_path

            try:
                if item.is_file() and self._should_include_file(item):
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    extracted_files += 1

            except Exception as e:
                logger.warning(f"Could not copy {item}: {e}")
                continue

        logger.info(f"Extracted {extracted_files} files from {lib_name}")
        return extracted_files

    def _should_include_file(self, file_path: Path, is_root: bool = False) -> bool:
        """Determine if a file should be included in extraction"""

        # Always include specific root files
        if is_root and file_path.name in self.ALWAYS_INCLUDE:
            return True

        # Include source files and key documentation
        allowed_extensions = {
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".json",
            ".md",
            ".yml",
            ".yaml",
            ".toml",
            ".rs",
            ".go",
            ".java",
            ".kt",
            ".cs",
            ".php",
            ".rb",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".swift",
            ".dart",
            ".scala",
            ".clj",
            ".hs",
        }

        if file_path.suffix.lower() in allowed_extensions:
            return True

        # Include files without extensions that might be important
        if not file_path.suffix and file_path.name in {
            "Dockerfile",
            "Makefile",
            "Rakefile",
            "Gemfile",
            "Pipfile",
        }:
            return True

        return False


def create_extraction_report(
    project_dir: Path,
    output_dir: Path,
    dependencies: Set[str],
    mappings: Dict[str, Path],
    extracted_count: int,
    total_files: int,
):
    """Create a report of what was extracted"""
    report_path = output_dir / "EXTRACTION_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Generic Reference Code Extraction Report\n\n")
        f.write(
            f"Generated on: {subprocess.check_output(['date']).decode().strip()}\n\n"
        )
        f.write(f"**Project Directory**: `{project_dir}`\n\n")

        f.write("## Project Dependencies Found\n\n")
        f.write(f"Total dependencies discovered: **{len(dependencies)}**\n\n")

        # Group dependencies by type (basic heuristic)
        frontend_deps = {
            dep
            for dep in dependencies
            if any(char in dep for char in ["@", "react", "vue", "angular", "svelte"])
        }
        backend_deps = dependencies - frontend_deps

        if frontend_deps:
            f.write("### Frontend/JavaScript Dependencies\n")
            for dep in sorted(frontend_deps):
                f.write(f"- {dep}\n")
            f.write("\n")

        if backend_deps:
            f.write("### Backend/Other Dependencies\n")
            for dep in sorted(backend_deps):
                f.write(f"- {dep}\n")
            f.write("\n")

        f.write("## Extraction Results\n\n")
        f.write(f"- **Dependencies with reference repositories**: {len(mappings)}\n")
        f.write(f"- **Successfully extracted**: {extracted_count}\n")
        f.write(f"- **Total files extracted**: {total_files}\n\n")

        if mappings:
            f.write("### Extracted Libraries\n\n")
            for dep_name, source_path in sorted(mappings.items()):
                f.write(f"#### {dep_name}\n")
                f.write(f"- **Source**: `{source_path}`\n")
                f.write(f"- **Extracted to**: `{output_dir / dep_name}/`\n\n")

        # Dependencies without references
        missing_deps = dependencies - set(mappings.keys())
        if missing_deps:
            f.write(
                f"### Dependencies Without Reference Repositories ({len(missing_deps)})\n\n"
            )
            for dep in sorted(missing_deps):
                f.write(f"- {dep}\n")
            f.write("\n")

        f.write("## Usage\n\n")
        f.write(
            "This directory contains only the core source files from each library, "
        )
        f.write(
            "filtered to remove tests, documentation, examples, and build artifacts. "
        )
        f.write("Use this for reference when working on your project, or to provide ")
        f.write("context to LLMs without including unnecessary files.\n\n")

        f.write("To regenerate this extraction:\n")
        f.write("```bash\n")
        f.write(
            f"python extract_reference_code_generic.py --project-dir {project_dir} --clean\n"
        )
        f.write("```\n")

    logger.info(f"Created extraction report: {report_path}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract relevant reference library code for any project"
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory of your project (default: current directory)",
    )
    parser.add_argument(
        "--references-dir",
        type=Path,
        default=Path.cwd() / "reference_repos",
        help="Directory containing reference git repositories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "extracted_reference_code",
        help="Output directory for extracted code",
    )
    parser.add_argument(
        "--update-refs", action="store_true", help="Update git repositories first"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove existing output directory"
    )
    parser.add_argument(
        "--config-file", type=Path, help="Custom configuration file (JSON)"
    )
    args = parser.parse_args()

    # Validate directories
    if not args.project_dir.exists():
        logger.error(f"Project directory does not exist: {args.project_dir}")
        sys.exit(1)

    if not args.references_dir.exists():
        logger.error(f"References directory does not exist: {args.references_dir}")
        logger.info(
            "Create it and add git repositories of libraries you want to reference"
        )
        sys.exit(1)

    # Clean output directory if requested
    if args.clean and args.output_dir.exists():
        logger.info(f"Removing existing {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # Update git repositories if requested
    if args.update_refs:
        logger.info("Updating git repositories...")
        for item in args.references_dir.rglob(".git"):
            if item.is_dir():
                repo_dir = item.parent
                try:
                    subprocess.run(
                        ["git", "fetch", "--all"],
                        cwd=repo_dir,
                        check=True,
                        capture_output=True,
                    )
                    logger.info(f"Updated {repo_dir}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Could not update {repo_dir}: {e}")

    # Analyze project dependencies
    logger.info(f"Analyzing dependencies in {args.project_dir}")
    analyzer = GenericDependencyAnalyzer(args.project_dir)
    dependencies = analyzer.analyze_all_dependencies()

    if not dependencies:
        logger.warning("No dependencies found in project")
        print("❌ No dependencies found. Make sure you're in a project directory with:")
        print("   - package.json (JavaScript/Node)")
        print("   - pyproject.toml or requirements.txt (Python)")
        print("   - Cargo.toml (Rust)")
        print("   - go.mod (Go)")
        print("   - etc.")
        sys.exit(1)

    # Discover reference repositories
    logger.info(f"Discovering reference repositories in {args.references_dir}")
    discovery = GenericReferenceDiscovery(args.references_dir, dependencies)
    mappings = discovery.discover_references()

    if not mappings:
        logger.warning("No reference repositories found for project dependencies")
        print("❌ No matching reference repositories found.")
        print(f"   Add git repositories to {args.references_dir}")
        print("   Repository names should match dependency names or be similar")
        sys.exit(1)

    # Extract code
    logger.info(f"Extracting code to {args.output_dir}")
    extractor = GenericCodeExtractor(args.output_dir)
    extracted_count, total_files = extractor.extract_all(mappings)

    # Create report
    create_extraction_report(
        args.project_dir,
        args.output_dir,
        dependencies,
        mappings,
        extracted_count,
        total_files,
    )

    # Show summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Project directory: {args.project_dir}")
    print(f"Dependencies found: {len(dependencies)}")
    print(f"Reference repositories: {len(mappings)}")
    print(f"Successfully extracted: {extracted_count}")
    print(f"Total files extracted: {total_files}")
    print(f"Output directory: {args.output_dir}")
    print(f"Report: {args.output_dir}/EXTRACTION_REPORT.md")

    if len(mappings) < len(dependencies):
        missing = len(dependencies) - len(mappings)
        print(f"\n⚠️  {missing} dependencies don't have reference repositories")
        print(f"   Add them to {args.references_dir} if needed")


if __name__ == "__main__":
    main()
