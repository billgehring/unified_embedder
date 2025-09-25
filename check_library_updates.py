#!/usr/bin/env python3
"""
Library Update Checker

Checks for available updates and potential breaking changes before updating
your reference libraries. Helps you make informed decisions about when to update.

Usage:
    python check_library_updates.py [--check-breaking] [--format json|table]
"""

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
GITHUB_CLONES_DIR = PROJECT_ROOT / "github_clones_for_reference"


@dataclass
class LibraryInfo:
    name: str
    current_version: str
    latest_version: str
    is_major_update: bool
    is_minor_update: bool
    changelog_url: Optional[str]
    breaking_changes: List[str]
    repo_path: Path


class BreakingChangeDetector:
    """Detect potential breaking changes from changelogs and version patterns"""

    BREAKING_PATTERNS = [
        r"breaking change",
        r"breaking:",
        r"BREAKING:",
        r"backwards? incompatible",
        r"removed.*api",
        r"deprecated.*removed",
        r"requires? migration",
        r"major.*update",
        r"migration guide",
        r"upgrade guide",
    ]

    def __init__(self):
        self.breaking_regex = re.compile(
            "|".join(self.BREAKING_PATTERNS), re.IGNORECASE
        )

    def check_changelog(
        self, repo_path: Path, from_version: str, to_version: str
    ) -> List[str]:
        """Check changelog files for breaking changes between versions"""
        breaking_changes = []

        changelog_files = [
            "CHANGELOG.md",
            "CHANGELOG.rst",
            "CHANGELOG.txt",
            "HISTORY.md",
            "RELEASES.md",
            "NEWS.md",
        ]

        for changelog_file in changelog_files:
            changelog_path = repo_path / changelog_file
            if changelog_path.exists():
                try:
                    with open(changelog_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Look for version sections and breaking changes
                    changes = self._extract_version_changes(
                        content, from_version, to_version
                    )
                    breaking_changes.extend(changes)

                except Exception as e:
                    logger.warning(f"Could not read {changelog_path}: {e}")

        return breaking_changes

    def _extract_version_changes(
        self, content: str, from_version: str, to_version: str
    ) -> List[str]:
        """Extract changes between versions from changelog content"""
        changes = []
        lines = content.split("\n")

        # Simple approach: look for breaking change patterns in recent entries
        in_recent_section = False
        current_section = ""

        for line in lines:
            # Version header detection (various formats)
            if re.match(r"^#+\s*\[?v?\d+\.\d+", line) or re.match(
                r"^#+\s*\d+\.\d+", line
            ):
                current_section = line.strip()
                # Simple heuristic: if this looks like a recent version, include it
                if any(v in line for v in [to_version, from_version]):
                    in_recent_section = True
                else:
                    in_recent_section = False

            # Look for breaking change indicators
            if in_recent_section and self.breaking_regex.search(line):
                changes.append(f"{current_section}: {line.strip()}")

        return changes


class LibraryUpdateChecker:
    """Check for updates to libraries in github_clones_for_reference"""

    def __init__(self):
        self.breaking_detector = BreakingChangeDetector()
        self.libraries: Dict[str, LibraryInfo] = {}

    def get_git_version(self, repo_path: Path) -> Tuple[str, str]:
        """Get current and latest git version for a repository"""
        try:
            # Get current commit/tag
            current = (
                subprocess.check_output(
                    ["git", "describe", "--tags", "--abbrev=0"],
                    cwd=repo_path,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            try:
                # Fallback to commit hash if no tags
                current = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"], cwd=repo_path
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                current = "unknown"

        try:
            # Fetch latest and get remote version
            subprocess.check_output(
                ["git", "fetch", "--tags"], cwd=repo_path, stderr=subprocess.DEVNULL
            )
            latest = (
                subprocess.check_output(
                    ["git", "describe", "--tags", "--abbrev=0", "origin/HEAD"],
                    cwd=repo_path,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            latest = current  # Assume no updates if can't fetch

        return current, latest

    def get_npm_version(self, package_name: str) -> Tuple[str, str]:
        """Get current and latest npm version"""
        try:
            # Get latest version from npm registry
            response = requests.get(
                f"https://registry.npmjs.org/{package_name}/latest", timeout=10
            )
            if response.status_code == 200:
                latest = response.json().get("version", "unknown")
            else:
                latest = "unknown"

            # For current version, we'd need to check package.json in the repo
            # For now, return unknown
            current = "unknown"

        except Exception as e:
            logger.warning(f"Could not check npm version for {package_name}: {e}")
            current = latest = "unknown"

        return current, latest

    def get_pypi_version(self, package_name: str) -> Tuple[str, str]:
        """Get current and latest PyPI version"""
        try:
            response = requests.get(
                f"https://pypi.org/pypi/{package_name}/json", timeout=10
            )
            if response.status_code == 200:
                latest = response.json()["info"]["version"]
            else:
                latest = "unknown"

            current = "unknown"  # Would need to parse from repo

        except Exception as e:
            logger.warning(f"Could not check PyPI version for {package_name}: {e}")
            current = latest = "unknown"

        return current, latest

    def is_major_version_change(self, current: str, latest: str) -> bool:
        """Check if version change is a major version bump"""
        try:
            current_parts = current.lstrip("v").split(".")
            latest_parts = latest.lstrip("v").split(".")

            if len(current_parts) >= 1 and len(latest_parts) >= 1:
                return int(current_parts[0]) < int(latest_parts[0])
        except (ValueError, IndexError):
            pass
        return False

    def is_minor_version_change(self, current: str, latest: str) -> bool:
        """Check if version change is a minor version bump"""
        try:
            current_parts = current.lstrip("v").split(".")
            latest_parts = latest.lstrip("v").split(".")

            if len(current_parts) >= 2 and len(latest_parts) >= 2:
                return int(current_parts[0]) == int(latest_parts[0]) and int(
                    current_parts[1]
                ) < int(latest_parts[1])
        except (ValueError, IndexError):
            pass
        return False

    def check_all_libraries(
        self, check_breaking: bool = False
    ) -> Dict[str, LibraryInfo]:
        """Check all libraries in github_clones_for_reference for updates"""

        if not GITHUB_CLONES_DIR.exists():
            logger.error(f"Directory {GITHUB_CLONES_DIR} does not exist")
            return {}

        for item in GITHUB_CLONES_DIR.iterdir():
            if item.is_dir() and item.name != ".git":
                self._check_library(item, check_breaking)

        return self.libraries

    def _check_library(self, repo_path: Path, check_breaking: bool):
        """Check a single library for updates"""
        lib_name = repo_path.name

        logger.info(f"Checking {lib_name}...")

        # Get version info (prefer git tags)
        current_version, latest_version = self.get_git_version(repo_path)

        # Check for breaking changes if requested
        breaking_changes = []
        if check_breaking and current_version != latest_version:
            breaking_changes = self.breaking_detector.check_changelog(
                repo_path, current_version, latest_version
            )

        # Create library info
        lib_info = LibraryInfo(
            name=lib_name,
            current_version=current_version,
            latest_version=latest_version,
            is_major_update=self.is_major_version_change(
                current_version, latest_version
            ),
            is_minor_update=self.is_minor_version_change(
                current_version, latest_version
            ),
            changelog_url=self._get_changelog_url(repo_path),
            breaking_changes=breaking_changes,
            repo_path=repo_path,
        )

        self.libraries[lib_name] = lib_info

    def _get_changelog_url(self, repo_path: Path) -> Optional[str]:
        """Get URL to changelog if it exists"""
        try:
            # Get remote URL
            remote_url = (
                subprocess.check_output(
                    ["git", "remote", "get-url", "origin"], cwd=repo_path
                )
                .decode()
                .strip()
            )

            # Convert to GitHub changelog URL if it's a GitHub repo
            if "github.com" in remote_url:
                # Convert git URL to HTTP and append changelog path
                http_url = remote_url.replace("git@github.com:", "https://github.com/")
                http_url = http_url.replace(".git", "")
                return f"{http_url}/blob/main/CHANGELOG.md"
        except subprocess.CalledProcessError:
            pass
        return None


def format_results(libraries: Dict[str, LibraryInfo], format_type: str = "table"):
    """Format results for display"""

    if format_type == "json":
        output = {}
        for name, lib in libraries.items():
            output[name] = {
                "current_version": lib.current_version,
                "latest_version": lib.latest_version,
                "has_updates": lib.current_version != lib.latest_version,
                "is_major_update": lib.is_major_update,
                "is_minor_update": lib.is_minor_update,
                "breaking_changes": lib.breaking_changes,
                "changelog_url": lib.changelog_url,
            }
        return json.dumps(output, indent=2)

    else:  # table format
        output = []
        output.append("=" * 80)
        output.append("LIBRARY UPDATE CHECK RESULTS")
        output.append("=" * 80)

        # Group by update status
        no_updates = []
        minor_updates = []
        major_updates = []
        unknown_updates = []

        for name, lib in libraries.items():
            if lib.current_version == lib.latest_version:
                no_updates.append(lib)
            elif lib.is_major_update:
                major_updates.append(lib)
            elif lib.is_minor_update:
                minor_updates.append(lib)
            else:
                unknown_updates.append(lib)

        # Major updates (potential breaking changes)
        if major_updates:
            output.append("\n🚨 MAJOR UPDATES AVAILABLE (Review carefully!)")
            output.append("-" * 50)
            for lib in major_updates:
                output.append(f"📦 {lib.name}")
                output.append(f"   Current: {lib.current_version}")
                output.append(f"   Latest:  {lib.latest_version}")
                if lib.breaking_changes:
                    output.append("   ⚠️  Breaking changes detected:")
                    for change in lib.breaking_changes[:3]:  # Show first 3
                        output.append(f"      • {change}")
                if lib.changelog_url:
                    output.append(f"   📖 Changelog: {lib.changelog_url}")
                output.append("")

        # Minor updates
        if minor_updates:
            output.append("\n✨ MINOR UPDATES AVAILABLE")
            output.append("-" * 30)
            for lib in minor_updates:
                output.append(
                    f"📦 {lib.name}: {lib.current_version} → {lib.latest_version}"
                )

        # Unknown/patch updates
        if unknown_updates:
            output.append("\n🔄 OTHER UPDATES AVAILABLE")
            output.append("-" * 25)
            for lib in unknown_updates:
                output.append(
                    f"📦 {lib.name}: {lib.current_version} → {lib.latest_version}"
                )

        # Up to date
        if no_updates:
            output.append(f"\n✅ UP TO DATE ({len(no_updates)} libraries)")
            output.append("-" * 15)
            for lib in no_updates:
                output.append(f"📦 {lib.name}: {lib.current_version}")

        # Summary
        output.append("\n📊 SUMMARY")
        output.append(f"   Total libraries: {len(libraries)}")
        output.append(f"   Major updates: {len(major_updates)}")
        output.append(f"   Minor updates: {len(minor_updates)}")
        output.append(f"   Other updates: {len(unknown_updates)}")
        output.append(f"   Up to date: {len(no_updates)}")

        return "\n".join(output)


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for library updates and breaking changes"
    )
    parser.add_argument(
        "--check-breaking",
        action="store_true",
        help="Check changelogs for breaking changes (slower)",
    )
    parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    args = parser.parse_args()

    logger.info("Starting library update check...")

    checker = LibraryUpdateChecker()
    libraries = checker.check_all_libraries(check_breaking=args.check_breaking)

    if not libraries:
        print("❌ No libraries found to check")
        sys.exit(1)

    # Format and display results
    results = format_results(libraries, args.format)
    print(results)

    # Exit with status code based on results
    major_updates = sum(1 for lib in libraries.values() if lib.is_major_update)
    if major_updates > 0:
        logger.warning(
            f"Found {major_updates} major updates that may contain breaking changes"
        )
        sys.exit(2)  # Exit code 2 for major updates available


if __name__ == "__main__":
    main()
