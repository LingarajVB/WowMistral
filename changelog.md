## 0.2.0 (2026-02-01)
### Added
- **Image Annotation**: AI-powered image classification with category, confidence, and reasoning
  - Uses Mistral's `bbox_annotation_format` with Pydantic schema
  - Optional interactive prompt when preserving images
  - Annotations appear as blockquotes below images in markdown
- Added `pydantic>=2.0.0` dependency for annotation schemas

### Fixed
- **Markdown Image Paths**: Wrapped paths in angle brackets `<>` to support filenames with spaces and special characters
  - Before: `![img](images/My PDF/page1.jpeg)` (broken)
  - After: `![img](<images/My PDF/page1.jpeg>)` (works)

---

## 0.1.0 (2026-01-28)
### Added
- Initial release
### Changed
- None
### Removed
- None

