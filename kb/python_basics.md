# Python Basics

## Variables and Data Types

Python is a dynamically typed language, meaning you don't need to declare variable types explicitly. Common data types include integers (int), floating-point numbers (float), strings (str), booleans (bool), lists, dictionaries, and tuples. Variables are created by assignment using the equals sign. For example, `name = "Alice"` creates a string variable, while `count = 42` creates an integer.

Python strings can be created using single quotes, double quotes, or triple quotes for multi-line strings. String formatting can be done using f-strings (f"Hello, {name}"), the format() method, or percent formatting. Strings are immutable, meaning once created they cannot be modified in place.

## Functions and Control Flow

Functions in Python are defined using the `def` keyword, followed by the function name and parameters in parentheses. Functions can return values using the `return` statement. If no return statement is present, the function returns None. Python supports default parameter values, keyword arguments, and variable-length argument lists using *args and **kwargs.

Control flow in Python uses if/elif/else statements for conditional execution, while and for loops for iteration, and try/except blocks for error handling. The for loop iterates over sequences like lists, strings, or ranges. Python uses indentation (typically 4 spaces) to define code blocks instead of curly braces.

## Modules and Packages

A module is a single Python file containing definitions and statements. Packages are directories containing multiple modules and an __init__.py file. You import modules using the `import` statement: `import math` or `from math import sqrt`. The `__name__` variable equals `"__main__"` when a script is run directly.

Virtual environments isolate project dependencies using `python -m venv .venv`. Activate with `source .venv/bin/activate` on Linux/Mac or `.\.venv\Scripts\Activate.ps1` on Windows. Install packages with pip: `pip install package-name`. Save dependencies with `pip freeze > requirements.txt` or use pyproject.toml for modern projects.

## Error Handling

Python uses exceptions to handle errors. Common exceptions include ValueError, TypeError, KeyError, and FileNotFoundError. Use try/except blocks to catch exceptions: the code in the try block is executed, and if an exception occurs, the except block handles it. You can catch specific exceptions or use a bare except to catch all.

The `finally` clause executes whether or not an exception occurred, useful for cleanup like closing files. The `with` statement provides automatic resource management: `with open('file.txt') as f:` ensures the file is closed even if an error occurs. Raise custom exceptions using `raise ValueError("message")`.
