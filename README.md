# markdown_quote

`markdown_quote` is a command-line tool that processes Markdown files to automatically replace special quote blocks with content extracted from referenced source files. It is especially useful for maintaining documentation with up-to-date code snippets.

## Quote Block Syntax

### Basic Format

```markdown
<!-- quote_begin content="[description](file_path#Lstart_line-Lend_line)" lang="language" -->
up-to-date content
<!-- quote_end -->
```

### Parameters

`content`: Specifies the file and line range to quote

- `description`: Optional description (currently unused)

- `file_path`: Relative path to the source file

- `#Lstart_line-Lend_line`: Line range (1-based inclusive).  If omitted, the start or end of the file is used.

Examples:

`L10-L20`: from line 10 to line 20

`L-L`: the entire file

`L10-L`: from line 10 to the end

`L-L30`: from the start to line 30

`lang`: Programming language for syntax highlighting

## Usage

Install `markdown_quote`
```
pip install markdown_quote
```

Navigate to your project folder and run:
```
markdown_quote
```
This command updates all quote blocks in Markdown files within the directory.
