"""Optional, dependency-heavy add-ons kept out of the lean core.

Subpackages here (e.g. ``extensions.trace``) are imported by the core only
through guarded try/except blocks, so the base runtime works without them.
"""
