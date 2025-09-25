#!/usr/bin/env python3
"""
Reference Libraries Update Workflow

Safe workflow for updating reference libraries with breaking change detection
and rollback capabilities.

Usage:
    python update_reference_libraries.py [--check-only] [--force] [--libs lib1,lib2]
"""

import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Import our other scripts
from check_library_updates import LibraryInfo, LibraryUpdateChecker
from extract_reference_code import main as extract_main

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
GITHUB_CLONES_DIR = PROJECT_ROOT / "github_clones_for_reference"
REFERENCE_CODE_DIR = PROJECT_ROOT / "reference_library_code"
BACKUP_DIR = PROJECT_ROOT / ".library_backups"


class SafeUpdateManager:
    """Manages safe updates with backup and rollback capabilities"""

    def __init__(self):
        self.backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_backup_dir = BACKUP_DIR / self.backup_timestamp
        self.updated_libraries: List[str] = []
        self.failed_libraries: List[str] = []

    def create_backup(self, library_names: List[str] = None):
        """Create backup of current state before updating"""
        logger.info("Creating backup before update...")

        self.current_backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup github_clones_for_reference (or specific libraries)
        if library_names:
            for lib_name in library_names:
                source_path = GITHUB_CLONES_DIR / lib_name
                if source_path.exists():
                    backup_path = (
                        self.current_backup_dir
                        / "github_clones_for_reference"
                        / lib_name
                    )
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_path, backup_path)
        else:
            backup_path = self.current_backup_dir / "github_clones_for_reference"
            shutil.copytree(GITHUB_CLONES_DIR, backup_path)

        # Backup current reference_library_code
        if REFERENCE_CODE_DIR.exists():
            backup_ref_path = self.current_backup_dir / "reference_library_code"
            shutil.copytree(REFERENCE_CODE_DIR, backup_ref_path)

        # Create backup manifest
        manifest = {
            "timestamp": self.backup_timestamp,
            "libraries": library_names or "all",
            "backup_path": str(self.current_backup_dir),
        }

        with open(self.current_backup_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Backup created at {self.current_backup_dir}")

    def update_library(self, lib_name: str, lib_info: LibraryInfo) -> bool:
        """Update a single library safely"""
        logger.info(f"Updating {lib_name}...")

        try:
            repo_path = lib_info.repo_path

            # Fetch latest changes
            subprocess.check_output(["git", "fetch", "--all"], cwd=repo_path)

            # Get the latest tag or main branch
            try:
                latest_ref = (
                    subprocess.check_output(
                        ["git", "describe", "--tags", "--abbrev=0", "origin/HEAD"],
                        cwd=repo_path,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                # Fallback to main/master branch
                latest_ref = "origin/main"
                try:
                    subprocess.check_output(
                        ["git", "rev-parse", "--verify", latest_ref], cwd=repo_path
                    )
                except subprocess.CalledProcessError:
                    latest_ref = "origin/master"

            # Checkout the latest version
            subprocess.check_output(["git", "checkout", latest_ref], cwd=repo_path)

            self.updated_libraries.append(lib_name)
            logger.info(f"✅ Updated {lib_name} to {latest_ref}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to update {lib_name}: {e}")
            self.failed_libraries.append(lib_name)
            return False

    def rollback(self, library_names: List[str] = None):
        """Rollback to the backup"""
        if not self.current_backup_dir.exists():
            logger.error("No backup found to rollback to")
            return False

        logger.info("Rolling back changes...")

        try:
            # Rollback github_clones_for_reference
            backup_clones = self.current_backup_dir / "github_clones_for_reference"
            if backup_clones.exists():
                if library_names:
                    for lib_name in library_names:
                        backup_lib = backup_clones / lib_name
                        current_lib = GITHUB_CLONES_DIR / lib_name
                        if backup_lib.exists() and current_lib.exists():
                            shutil.rmtree(current_lib)
                            shutil.copytree(backup_lib, current_lib)
                else:
                    if GITHUB_CLONES_DIR.exists():
                        shutil.rmtree(GITHUB_CLONES_DIR)
                    shutil.copytree(backup_clones, GITHUB_CLONES_DIR)

            # Rollback reference_library_code
            backup_ref = self.current_backup_dir / "reference_library_code"
            if backup_ref.exists():
                if REFERENCE_CODE_DIR.exists():
                    shutil.rmtree(REFERENCE_CODE_DIR)
                shutil.copytree(backup_ref, REFERENCE_CODE_DIR)

            logger.info("✅ Rollback completed")
            return True

        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False

    def cleanup_old_backups(self, keep_count: int = 5):
        """Clean up old backups, keeping only the most recent ones"""
        if not BACKUP_DIR.exists():
            return

        backups = sorted([d for d in BACKUP_DIR.iterdir() if d.is_dir()], reverse=True)

        for backup in backups[keep_count:]:
            logger.info(f"Cleaning up old backup: {backup}")
            shutil.rmtree(backup)


def get_user_confirmation(message: str) -> bool:
    """Get user confirmation for potentially risky actions"""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Safely update reference libraries")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for updates, don't perform them",
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--libs", type=str, help="Comma-separated list of specific libraries to update"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup (not recommended)",
    )
    args = parser.parse_args()

    # Parse specific libraries if provided
    specific_libs = None
    if args.libs:
        specific_libs = [lib.strip() for lib in args.libs.split(",")]

    # Check for updates
    logger.info("Checking for library updates...")
    checker = LibraryUpdateChecker()
    all_libraries = checker.check_all_libraries(check_breaking=True)

    if not all_libraries:
        print("❌ No libraries found to check")
        sys.exit(1)

    # Filter libraries that need updates
    libraries_to_update = {}
    for name, lib_info in all_libraries.items():
        if specific_libs and name not in specific_libs:
            continue
        if lib_info.current_version != lib_info.latest_version:
            libraries_to_update[name] = lib_info

    if not libraries_to_update:
        print("✅ All libraries are up to date!")
        return

    # Show update summary
    print(f"\n📋 UPDATES AVAILABLE ({len(libraries_to_update)} libraries)")
    print("=" * 50)

    major_updates = []
    minor_updates = []
    other_updates = []

    for name, lib_info in libraries_to_update.items():
        update_info = f"{name}: {lib_info.current_version} → {lib_info.latest_version}"

        if lib_info.is_major_update:
            major_updates.append((name, lib_info, update_info))
        elif lib_info.is_minor_update:
            minor_updates.append((name, lib_info, update_info))
        else:
            other_updates.append((name, lib_info, update_info))

    if major_updates:
        print("\n🚨 MAJOR UPDATES (may contain breaking changes):")
        for name, lib_info, update_info in major_updates:
            print(f"   {update_info}")
            if lib_info.breaking_changes:
                print("      ⚠️  Potential breaking changes detected!")

    if minor_updates:
        print("\n✨ MINOR UPDATES:")
        for name, lib_info, update_info in minor_updates:
            print(f"   {update_info}")

    if other_updates:
        print("\n🔄 OTHER UPDATES:")
        for name, lib_info, update_info in other_updates:
            print(f"   {update_info}")

    # Exit if check-only
    if args.check_only:
        print("\n✅ Check complete. Use without --check-only to perform updates.")
        return

    # Get confirmation for updates
    if not args.force:
        if major_updates:
            print("\n⚠️  WARNING: Major updates detected!")
            print(
                "   These may contain breaking changes that could affect your project."
            )
            if not get_user_confirmation("Continue with major updates?"):
                print("❌ Update cancelled")
                return

        if not get_user_confirmation(f"Update {len(libraries_to_update)} libraries?"):
            print("❌ Update cancelled")
            return

    # Perform updates
    update_manager = SafeUpdateManager()

    try:
        # Create backup
        if not args.no_backup:
            update_manager.create_backup(list(libraries_to_update.keys()))

        # Update libraries
        for name, lib_info in libraries_to_update.items():
            update_manager.update_library(name, lib_info)

        # Re-extract reference code
        logger.info("Re-extracting reference library code...")
        extract_main()

        # Show results
        print("\n📊 UPDATE RESULTS")
        print("=" * 30)
        print(f"✅ Successfully updated: {len(update_manager.updated_libraries)}")
        for lib in update_manager.updated_libraries:
            print(f"   • {lib}")

        if update_manager.failed_libraries:
            print(f"❌ Failed to update: {len(update_manager.failed_libraries)}")
            for lib in update_manager.failed_libraries:
                print(f"   • {lib}")

        if not args.no_backup:
            print(f"\n💾 Backup available at: {update_manager.current_backup_dir}")
            print("   Use --rollback if you need to revert changes")

        # Clean up old backups
        update_manager.cleanup_old_backups()

        print("\n✅ Update process completed!")

    except KeyboardInterrupt:
        print("\n⏹️  Update interrupted by user")
        if not args.no_backup and get_user_confirmation("Rollback to backup?"):
            update_manager.rollback(list(libraries_to_update.keys()))

    except Exception as e:
        logger.error(f"❌ Update process failed: {e}")
        if not args.no_backup:
            logger.info("Attempting automatic rollback...")
            update_manager.rollback(list(libraries_to_update.keys()))


if __name__ == "__main__":
    main()
