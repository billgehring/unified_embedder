#!/usr/bin/env python3
"""
Reference Libraries Management - Complete Workflow

Single script to manage your reference libraries with safety checks,
breaking change detection, and automated workflows.

Usage:
    python manage_reference_libraries.py [command] [options]

Commands:
    check          - Check for available updates
    update         - Update libraries (with safety checks)
    extract        - Re-extract reference code from current libraries
    status         - Show current status of all libraries
    schedule       - Set up automated update schedule
    rollback       - Rollback to a previous backup

Examples:
    python manage_reference_libraries.py check
    python manage_reference_libraries.py update --minor-only
    python manage_reference_libraries.py update --libs docling,drei
    python manage_reference_libraries.py rollback --list
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Import our components
from check_library_updates import LibraryUpdateChecker, format_results
from extract_reference_code import main as extract_main
from update_reference_libraries import SafeUpdateManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / ".reference_library_config.json"
BACKUP_DIR = PROJECT_ROOT / ".library_backups"


class LibraryManager:
    """Main library management class"""

    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "update_frequency": "monthly",
            "auto_minor_updates": False,
            "skip_major_updates": True,
            "notification_email": None,
            "exclude_libraries": [],
            "last_update_check": None,
            "update_schedule": "weekly_check",
        }

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        return default_config

    def save_config(self):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save config: {e}")

    def check_updates(self, check_breaking=True):
        """Check for updates and return formatted results"""
        checker = LibraryUpdateChecker()
        libraries = checker.check_all_libraries(check_breaking=check_breaking)

        # Update last check time
        self.config["last_update_check"] = datetime.now().isoformat()
        self.save_config()

        return libraries

    def should_auto_update(self, lib_info):
        """Determine if a library should be auto-updated based on config"""
        if lib_info.name in self.config.get("exclude_libraries", []):
            return False

        if lib_info.is_major_update and self.config.get("skip_major_updates", True):
            return False

        if lib_info.is_minor_update and self.config.get("auto_minor_updates", False):
            return True

        return False

    def update_libraries(self, specific_libs=None, minor_only=False, force=False):
        """Update libraries with safety checks"""
        libraries = self.check_updates()

        # Filter libraries to update
        libraries_to_update = {}
        for name, lib_info in libraries.items():
            if specific_libs and name not in specific_libs:
                continue
            if lib_info.current_version == lib_info.latest_version:
                continue
            if minor_only and lib_info.is_major_update:
                logger.info(
                    f"Skipping major update for {name} (--minor-only specified)"
                )
                continue

            libraries_to_update[name] = lib_info

        if not libraries_to_update:
            print("✅ No updates needed!")
            return True

        # Use the safe update manager
        from update_reference_libraries import main as update_main

        # Temporarily modify sys.argv to pass arguments to update script
        original_argv = sys.argv[:]
        sys.argv = ["update_reference_libraries.py"]

        if force:
            sys.argv.append("--force")
        if specific_libs:
            sys.argv.extend(["--libs", ",".join(specific_libs)])

        try:
            update_main()
            return True
        except SystemExit as e:
            return e.code == 0
        finally:
            sys.argv = original_argv

    def extract_reference_code(self):
        """Re-extract reference code"""
        logger.info("Extracting reference library code...")
        extract_main()
        print("✅ Reference code extraction completed!")

    def show_status(self):
        """Show current status of all libraries"""
        libraries = self.check_updates(check_breaking=False)

        print("📊 REFERENCE LIBRARIES STATUS")
        print("=" * 50)

        # Show config
        print("\n⚙️  Configuration:")
        print(f"   Update frequency: {self.config.get('update_frequency', 'manual')}")
        print(f"   Auto minor updates: {self.config.get('auto_minor_updates', False)}")
        print(f"   Skip major updates: {self.config.get('skip_major_updates', True)}")
        last_check = self.config.get("last_update_check")
        if last_check:
            print(f"   Last check: {last_check}")

        # Show library status
        print(format_results(libraries, "table"))

        # Show recent backups
        if BACKUP_DIR.exists():
            backups = sorted(
                [d for d in BACKUP_DIR.iterdir() if d.is_dir()], reverse=True
            )
            if backups:
                print(f"\n💾 Recent backups ({len(backups)} available):")
                for backup in backups[:3]:
                    print(f"   • {backup.name}")
                if len(backups) > 3:
                    print(f"   ... and {len(backups) - 3} more")

    def list_backups(self):
        """List available backups"""
        if not BACKUP_DIR.exists():
            print("📂 No backups found")
            return []

        backups = sorted([d for d in BACKUP_DIR.iterdir() if d.is_dir()], reverse=True)

        print("💾 AVAILABLE BACKUPS")
        print("=" * 30)

        for i, backup in enumerate(backups):
            manifest_file = backup / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file) as f:
                        manifest = json.load(f)
                    libraries = manifest.get("libraries", "unknown")
                    if isinstance(libraries, list):
                        lib_count = len(libraries)
                        lib_text = f"{lib_count} libraries"
                    else:
                        lib_text = str(libraries)
                    print(f"{i+1:2d}. {backup.name} - {lib_text}")
                except Exception:
                    print(f"{i+1:2d}. {backup.name} - (no manifest)")
            else:
                print(f"{i+1:2d}. {backup.name} - (no manifest)")

        return backups

    def rollback_to_backup(self, backup_name=None):
        """Rollback to a specific backup"""
        backups = self.list_backups()

        if not backups:
            print("❌ No backups available for rollback")
            return False

        if backup_name:
            # Find backup by name
            backup_dir = None
            for backup in backups:
                if backup.name == backup_name:
                    backup_dir = backup
                    break

            if not backup_dir:
                print(f"❌ Backup '{backup_name}' not found")
                return False
        else:
            # Interactive selection
            print("\nSelect a backup to rollback to:")
            try:
                choice = int(input("Enter backup number: ")) - 1
                if 0 <= choice < len(backups):
                    backup_dir = backups[choice]
                else:
                    print("❌ Invalid selection")
                    return False
            except (ValueError, KeyboardInterrupt):
                print("❌ Rollback cancelled")
                return False

        # Confirm rollback
        if (
            not input(
                f"⚠️  Rollback to {backup_dir.name}? This will overwrite current libraries (y/n): "
            )
            .lower()
            .startswith("y")
        ):
            print("❌ Rollback cancelled")
            return False

        # Perform rollback
        update_manager = SafeUpdateManager()
        update_manager.current_backup_dir = backup_dir

        if update_manager.rollback():
            print("✅ Rollback completed successfully!")
            # Re-extract reference code
            self.extract_reference_code()
            return True
        else:
            print("❌ Rollback failed!")
            return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Manage reference libraries for SlideQuest project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check                    # Check for updates
  %(prog)s update --minor-only      # Update only minor versions
  %(prog)s update --libs drei,docling  # Update specific libraries
  %(prog)s status                   # Show current status
  %(prog)s rollback --list          # List available backups
  %(prog)s extract                  # Re-extract reference code
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check for library updates")
    check_parser.add_argument(
        "--no-breaking",
        action="store_true",
        help="Skip breaking change analysis (faster)",
    )
    check_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )

    # Update command
    update_parser = subparsers.add_parser("update", help="Update libraries")
    update_parser.add_argument(
        "--minor-only",
        action="store_true",
        help="Only update minor versions (skip major)",
    )
    update_parser.add_argument(
        "--libs", type=str, help="Comma-separated list of libraries to update"
    )
    update_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )
    update_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup (not recommended)",
    )

    # Status command
    subparsers.add_parser("status", help="Show current status")

    # Extract command
    subparsers.add_parser("extract", help="Re-extract reference code")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to backup")
    rollback_parser.add_argument(
        "--list", action="store_true", help="List available backups"
    )
    rollback_parser.add_argument(
        "--backup", type=str, help="Specific backup name to rollback to"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create manager instance
    manager = LibraryManager()

    # Execute command
    try:
        if args.command == "check":
            libraries = manager.check_updates(check_breaking=not args.no_breaking)
            results = format_results(libraries, args.format)
            print(results)

        elif args.command == "update":
            specific_libs = None
            if args.libs:
                specific_libs = [lib.strip() for lib in args.libs.split(",")]

            success = manager.update_libraries(
                specific_libs=specific_libs,
                minor_only=args.minor_only,
                force=args.force,
            )

            if not success:
                sys.exit(1)

        elif args.command == "status":
            manager.show_status()

        elif args.command == "extract":
            manager.extract_reference_code()

        elif args.command == "rollback":
            if args.list:
                manager.list_backups()
            else:
                success = manager.rollback_to_backup(args.backup)
                if not success:
                    sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
