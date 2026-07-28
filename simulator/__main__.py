"""Allow `python -m mfu ...` to invoke the CLI in core.main."""

from .core import main

if __name__ == "__main__":
    main()
