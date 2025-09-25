# Reference Code Extraction Setup

This project is set up with automatic reference code extraction for development efficiency.

## Quick Start

```bash
# 1. Add reference repositories (one-time setup)
mkdir reference_repos
cd reference_repos
git clone https://github.com/your-library/repo.git
cd ..

# 2. Extract reference code
python extract_reference_code_generic.py

# 3. Check for updates
python manage_reference_libraries.py check

# 4. Update libraries safely
python manage_reference_libraries.py update --minor-only
```

## Directory Structure

```
your-project/
├── reference_repos/           # Git repositories of libraries you use
├── extracted_reference_code/  # Lightweight extracted source code
├── .library_backups/         # Automatic backups (git-ignored)
└── .reference_extraction_config.json  # Configuration
```

## Benefits

- **Lightweight**: Only source code, no tests/docs/examples
- **Up-to-date**: Automatic update checking and safe updating
- **Shareable**: Easy to share 18MB instead of 775MB+
- **LLM-friendly**: Perfect for providing context to AI assistants

## Usage Examples

```bash
# Check what libraries need updates
python manage_reference_libraries.py status

# Update specific libraries
python manage_reference_libraries.py update --libs fastapi,react

# Rollback if something breaks
python manage_reference_libraries.py rollback --list
python manage_reference_libraries.py rollback

# Re-extract after manual changes
python manage_reference_libraries.py extract
```

## Configuration

Edit `.reference_extraction_config.json` to customize:
- Which file types to include
- Directories to exclude
- Auto-update preferences

## Adding New Libraries

1. Clone the library to `reference_repos/`:
   ```bash
   cd reference_repos
   git clone https://github.com/library/repo.git
   ```

2. Re-extract:
   ```bash
   python extract_reference_code_generic.py --clean
   ```

The system will automatically detect the new library and include it.
