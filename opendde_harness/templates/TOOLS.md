# Tool Usage Notes

Tool signatures arrive through function calling. This file documents constraints that are not visible in the signatures.

## bash
- Commands time out (default 60 s); destructive commands (rm -rf, dd, shutdown, ...) are blocked; output is truncated at 10,000 characters.
- Long computations belong in design tasks, not in bash.

## read
- Images are returned as pictures and downscaled when large; crop the region you need instead of reading a whole screenshot.
- An image is visible only for the turn that read it; read it again if you need another look.
