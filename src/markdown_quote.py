"""
A command line tool that processes markdown files to replace quote blocks with actual content from referenced files.

Inspired by Jetbrains MarkdownQuote plugin:
https://plugins.jetbrains.com/plugin/22311-markdownquote

This tool scans markdown files for special quote blocks in the format:
<!-- quote_begin content="[description](file_path#Lstart_line-Lend_line)" lang="language" -->
...existing content...
<!-- quote_end -->

And replaces them with the actual content from the referenced files.
"""
import sys
from collections import deque, defaultdict
import os
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Regular expression patterns for parsing quote blocks
CONTENT_PATTERN = r'\s+content="\[([\s\S]*?)\]\((?P<content>[\s\S]*?)\)"'
LANG_PATTERN = r'\s+lang="(?P<lang>.*?)"'
VALUES_PATTERN = rf'(({CONTENT_PATTERN})|({LANG_PATTERN}))*'
BEGIN_PATTERN = rf'<!--\s*quote_begin({VALUES_PATTERN})\s*-->'
END_PATTERN = r'<!--\s*quote_end\s*-->'
QUOTE_PATTERN = rf'(?P<begin_block>{BEGIN_PATTERN})([.\s\S]*)(?P<end_block>{END_PATTERN})'


class MarkdownQuoteProcessor:
    """
    Processor for Markdown files that replaces quote blocks with content
    from referenced source files. Handles nested quote blocks correctly.
    """

    def __init__(self):
        # Patterns for individual tags (not for matching entire blocks)
        self.begin_pattern = re.compile(BEGIN_PATTERN)
        self.end_pattern = re.compile(END_PATTERN)

    def find_quote_blocks(self, content: str, file_name:str) -> list[Any]:
        """
        Find all quote blocks in content, handling nesting correctly.

        Args:
            content: Markdown content to parse
            file_name: Markdown file name

        Returns:
            List of quote block dictionaries with start, end, and parameters
        """
        blocks = []
        stack = []

        # Find all begin and end tags
        for begin_match in self.begin_pattern.finditer(content):
            stack.append({
                'type': 'begin',
                'match': begin_match,
                'params_str': begin_match.group(1),
                'start_pos': begin_match.start(),
                'end_pos': begin_match.end()
            })

        for end_match in self.end_pattern.finditer(content):
            stack.append({
                'type': 'end',
                'match': end_match,
                'start_pos': end_match.start(),
                'end_pos': end_match.end()
            })

        # Sort all tags by position
        stack.sort(key=lambda x: x['start_pos'])

        # Process stack to find matching begin-end pairs
        open_blocks = []
        for item in stack:

            if item['type'] == 'begin':
                open_blocks.append(item)
            else:  # end
                if open_blocks:
                    begin_item = open_blocks.pop()
                    blocks.append({
                        'begin_match': begin_item['match'],
                        'end_match': item['match'],
                        'start_pos': begin_item['start_pos'],
                        'end_pos': item['end_pos'],
                        'params_str': begin_item['params_str'],
                        'content_start': begin_item['end_pos'],
                        'content_end': item['start_pos']
                    })
        if len(stack) % 2 == 1 or len(open_blocks) != 0:
            print("Error parse quote blocks in :", file_name, file=sys.stderr)
            print("The stack:", file=sys.stderr)
            for (i, item) in enumerate(stack):
                print("|--> ", i, ". \n" , indent_text(content[item['start_pos']:item['end_pos']]),  file=sys.stderr)
            if len(open_blocks) != 0:
                print("The open blocks in the stack:", file=sys.stderr)
                for (i, item) in enumerate(open_blocks):
                    print("|--> ", i, ". \n" , indent_text(content[item['start_pos']:item['end_pos']]), file=sys.stderr)
            return []

        return blocks

    def pre_process_markdown_content(self, content: str, md_file_path: Path, dependency_map:Dict[str, Any]):
        """
        Process markdown content and replace all quote blocks.

        Args:
            dependency_map: dependency map
            content: Markdown content to process
            md_file_path: Markdown file path

        Returns:
            Processed content with quote blocks replaced
        """
        # Find all quote blocks
        blocks = self.find_quote_blocks(content, md_file_path.name)

        if not blocks:
            return

        # Process blocks from innermost to outermost to handle nesting
        # Sort by start position in reverse order so we replace from the end
        blocks.sort(key=lambda x: x['start_pos'], reverse=True)

        result = content
        for block in blocks:
            pre_process_quote_block(block, md_file_path, result, dependency_map)

    def process_markdown_content(self, content: str, file_path: Path) -> str:
        """
        Process markdown content and replace all quote blocks.

        Args:
            content: Markdown content to process
            file_path: Markdown file paths

        Returns:
            Processed content with quote blocks replaced
        """

        base_dir = file_path.parent
        # Find all quote blocks
        blocks = self.find_quote_blocks(content, file_path.name)

        if not blocks:
            return content

        range_blocks = []
        for block in blocks:
            range_blocks.append((block['start_pos'], block['end_pos'], block))

        # Process only outer blocks
        outer_blocks = filter_outer_ranges(range_blocks)

        # Sort by start position so we replace from the start
        outer_blocks.sort(key=lambda x: x[0])

        result = ""
        for i, (start_pos, end_pos, block) in enumerate(outer_blocks):
            # Replace the entire block (from begin tag to end tag)
            replacement = process_quote_block(block, base_dir)

            if i == 0:
                before = content[:start_pos]
                result += before
            else:
                prev_end_pos = outer_blocks[i - 1][1]
                before = content[prev_end_pos:start_pos]
                result += before
            result += replacement
            if i == len(outer_blocks) - 1:
                after = content[end_pos:]
                result += after


        return result

    def pre_process_markdown_file(self, file_path: Path, dependency_map:Dict[str, Any]):
        """
        Pre-Process a single Markdown file, retrieve dependencies.

        Args:
            file_path: Path to the Markdown file
            dependency_map: dependency map
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.pre_process_markdown_content(content, file_path, dependency_map)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def process_markdown_file(self, file_path: Path) -> bool:
        """
        Process a single Markdown file, replacing all quote blocks.

        Args:
            file_path: Path to the Markdown file

        Returns:
            True if file was modified, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get base directory for relative file paths (same as Markdown file directory)

            # Process content
            new_content = self.process_markdown_content(content, file_path)

            # Write back only if content changed
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True

            return False

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

    def process_directory(self, directory: Path) -> Tuple[int, int]:
        """
        Process all Markdown files in a directory.

        Args:
            directory: Directory to process

        Returns:
            Tuple of (files_processed, files_modified)
        """
        markdown_files = find_markdown_files(directory)

        processed_count = 0
        modified_count = 0

        dependencies_map = {}
        for file_path in markdown_files:
            self.pre_process_markdown_file(file_path, dependencies_map)

        sorted_files = topological_sort(dependencies_map)

        for file_path in sorted_files:
            if is_md_file(file_path):
                modified = self.process_markdown_file(Path(file_path).resolve())
                if modified:
                    modified_count += 1
                    print(f"Updated: {file_path}")
                processed_count += 1

        return processed_count, modified_count


def filter_outer_ranges(ranges):
    """
    Find ranges that are not contained within any other range.

    Args:
        ranges: List of tuples/lists representing ranges [m, n] where m <= n

    Returns:
        List of ranges that are not contained within any other range
    """
    if not ranges:
        return []

    # Sort ranges by start value, then by end value (descending)
    # This ensures larger ranges come first when start values are equal
    sorted_ranges = sorted(ranges, key=lambda x: (x[0], -x[1]))

    result = []
    max_end = float('-inf')

    for current_range in sorted_ranges:
        start, end, _ = current_range

        # If current range's end is greater than max_end seen so far,
        # it means this range is not contained within any previous range
        if end > max_end:
            result.append(current_range)
            max_end = end

    return result

def indent_text(multiline_text:str) -> str:
    text = ""
    indent = "    "
    for line in multiline_text.split('\n'):
        text += f"{indent}{line}\n"
    return text

def parse_content_parameter(content_str: str) -> Tuple[Optional[str], str, Optional[Tuple[int, int]]]:
    """
    Parse the content parameter to extract file path and line range.

    Args:
        content_str: Content parameter string in format [description](file_path#Lstart-Lend)

    Returns:
        Tuple of (description, file_path, line_range)
    """
    # Match the pattern [description](file_path#Lstart-Lend)
    match = re.match(r'\[(.*?)\]\((.*?)(#L(\d*)-L(\d*))?\)', content_str)
    if not match:
        raise ValueError(f"Invalid content format: {content_str}")

    description = match.group(1) if match.group(1) else None
    file_path = match.group(2)

    # Parse line range if present
    line_range = None
    if match.group(3):  # If line range is specified
        start_str = match.group(4)  # Could be empty
        end_str = match.group(5)  # Could be empty

        start_line = int(start_str) if start_str else None
        end_line = int(end_str) if end_str else None

        line_range = (start_line, end_line)

    return description, file_path, line_range


def extract_file_content(file_path: Path, line_range: Optional[Tuple[int, int]] = None) -> str:
    """
    Extract content from source file based on line range.

    Args:
        file_path: Path to the source file
        line_range: Optional tuple (start_line, end_line) - 1-based inclusive

    Returns:
        Extracted content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Handle different line range cases
        if line_range is None:
            # No line range specified - return entire file
            return ''.join(lines)

        start_line, end_line = line_range

        # Convert to 0-based indexing for list access
        start_idx = start_line - 1 if start_line is not None else 0
        end_idx = end_line if end_line is not None else len(lines)

        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(lines)))
        end_idx = max(0, min(end_idx, len(lines)))

        # Extract the specified lines
        extracted_lines = lines[start_idx:end_idx]
        return ''.join(extracted_lines)

    except FileNotFoundError:
        return f"<!-- ERROR: File not found: {file_path} -->"
    except Exception as e:
        return f"<!-- ERROR: Could not read file {file_path}: {str(e)} -->"


def parse_parameters(params_str: str) -> dict:
    """
    Parse parameters from quote_begin tag.

    Args:
        params_str: String containing parameters in key="value" format

    Returns:
        Dictionary of parsed parameters
    """
    params = {}
    params_pattern = re.compile(r'(\w+)="([^"]*)"')
    matches = params_pattern.findall(params_str)
    for key, value in matches:
        params[key] = value
    return params


def pre_process_quote_block(block: Dict[str, Any], md_file_path: Path, content: str, dependency_map:Dict[str, Any]):
    """
    Pre-Process a single quote block.

    Args:
        md_file_path: Markdown file path
        block: Quote block dictionary
        content: Original content (for extracting existing content)
        dependency_map: dependency map

    Returns:
        Replacement string for the quote block
    """
    params_str = block['params_str']
    original_content = content[block['content_start']:block['content_end']]

    try:
        # Parse parameters
        params = parse_parameters(params_str)

        # Extract content parameter details
        if 'content' not in params:
            return f"<!-- ERROR: Missing 'content' parameter -->{original_content}<!-- quote_end -->"

        _, file_path, _ = parse_content_parameter(params['content'])
        dir_path = md_file_path.parent
        full_file_path = to_full_path(file_path, dir_path)
        depend_file_normalized = normalized_path(full_file_path)
        # Get normalized path of current file
        this_file_normalized = normalized_path(md_file_path)
        # Add dependency relationship: depend_file -> this_file
        # Meaning: depend_file must be processed before this_file
        if depend_file_normalized not in dependency_map:
            dependency_map[depend_file_normalized] = {this_file_normalized}
        else:
            dependency_map[depend_file_normalized].add(this_file_normalized)
    except Exception as e:
        return f"<!-- ERROR: when pre processing {str(e)} -->{original_content}<!-- quote_end -->"


def process_quote_block(block: Dict[str, Any], base_dir: Path) -> str:
    """
        Process a single quote block. Keep the quote tags, only replace the content between them.

        Args:
            block: Quote block dictionary
            base_dir: Base directory for resolving relative file paths

        Returns:
            Replacement string that keeps the original tags but replaces the content
    """
    params_str = block['params_str']

    try:
        # Parse parameters
        params = parse_parameters(params_str)

        # Extract content parameter details
        if 'content' not in params:
            # Keep the original structure but mark error
            begin_tag = block['begin_match'].group(0)
            end_tag = block['end_match'].group(0)
            return f"{begin_tag}<!-- ERROR: Missing 'content' parameter -->{end_tag}"

        description, file_path, line_range = parse_content_parameter(params['content'])

        # Resolve file path relative to base directory
        full_file_path = base_dir / file_path

        # Extract content from source file
        extracted_content = extract_file_content(full_file_path, line_range)

        # Get language for code block if specified
        lang = params.get('lang', '')

        # Remove trailing newlines to avoid extra blank lines
        extracted_content = extracted_content.rstrip('\n')

        # Format the extracted content
        if lang:
            formatted_content = f"```{lang}\n{extracted_content}\n```"
        else:
            formatted_content = extracted_content

        # Keep the original quote tags, only replace the content between them
        begin_tag = block['begin_match'].group(0)
        end_tag = block['end_match'].group(0)
        return f"{begin_tag}\n{formatted_content}\n{end_tag}"

    except Exception as e:
        # Keep the original structure but include error message
        begin_tag = block['begin_match'].group(0)
        end_tag = block['end_match'].group(0)
        return f"{begin_tag}<!-- ERROR: {str(e)} -->{end_tag}"


def find_markdown_files(directory: Path) -> List[Path]:
    """
    Find all Markdown files in directory and subdirectories.

    Args:
        directory: Root directory to search

    Returns:
        List of paths to Markdown files
    """
    markdown_files = []
    for pattern in ['*.md', '*.markdown']:
        markdown_files.extend(directory.rglob(pattern))
    return markdown_files

def topological_sort(dependencies):
    """
    Perform topological sorting on dependency relationships using Kahn's algorithm.

    This ensures files are processed in the correct order based on their dependencies.
    If file A references file B, then file B should be processed before file A.

    Args:
        dependencies: Dictionary where key is a file path, value is a set of files that depend on the key
                    Format: {file_path: {dependent_file1, dependent_file2, ...}}
                    This means the key file must be processed before all files in the value set

    Returns:
        list: Files in topological order (files with no dependencies first),
              or empty list if a cycle is detected
    """
    # Build in-degree count and adjacency list
    in_degree = defaultdict(int)  # Tracks how many dependencies each file has
    graph = defaultdict(list)     # Maps files to their dependents

    # Collect all nodes from the dependency graph
    all_nodes = set()
    for key, dependents in dependencies.items():
        all_nodes.add(key)
        all_nodes.update(dependents)

    # Build the graph structure
    # If file A depends on file B, we have: B -> A (B must be processed before A)
    for key, dependents in dependencies.items():
        for dependent in dependents:
            graph[key].append(dependent)
            in_degree[dependent] += 1

    # Initialize queue with nodes having zero in-degree (no dependencies)
    queue = deque()
    for node in all_nodes:
        if in_degree[node] == 0:
            queue.append(node)

    # Perform topological sort using Kahn's algorithm
    result = []
    while queue:
        current_node = queue.popleft()
        result.append(current_node)

        # Process all nodes that depend on the current node
        for neighbor in graph.get(current_node, []):
            in_degree[neighbor] -= 1
            # If neighbor has no more dependencies, add to queue
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles - if result doesn't include all nodes, a cycle exists
    if len(result) != len(all_nodes):
        print("Cycle detected in dependency graph")
        return []

    return result



def to_full_path(file_path, md_file_dir):
    """
    Convert relative path to absolute path based on markdown file's directory.

    Args:
        file_path: Relative or absolute file path
        md_file_dir: Directory of the markdown file containing the reference

    Returns:
        str: Absolute file path
    """
    if os.path.isabs(file_path):
        return file_path
    else:
        return os.path.join(md_file_dir, file_path)

def normalized_path(file_path):
    """
    Normalize file path to absolute and standardized format.

    Args:
        file_path: Input file path

    Returns:
        str: Normalized absolute path
    """
    abs_path = os.path.abspath(file_path)
    return os.path.normpath(abs_path)

def is_md_file(filename):
    """
    Check if a file is a markdown file based on extension.

    Args:
        filename: File name or path to check

    Returns:
        bool: True if file has .md extension (case-insensitive)
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() == '.md'


def main():
    """
    Main function: Process all .md files in specified directory with dependency resolution.

    The process involves two passes:
    1. Pre-processing: Build dependency graph between files
    2. Processing: Process files in topological order based on dependencies
    """

    parser = argparse.ArgumentParser(description="markdown_quote processes markdown files to replace quote blocks with actual content from referenced files.")
    parser.add_argument('--version', action='version', version='0.0.1')
    parser.add_argument('--input', help="Input folder path to scan")

      # Directory to scan for markdown files
    args = parser.parse_args()
    folder_path = args.input
    if folder_path is None:
        folder_path = '.'

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return -1

    processor = MarkdownQuoteProcessor()
    processed_count, modified_count = processor.process_directory(Path(folder_path).resolve())

    print(f"Processed {processed_count} files, modified {modified_count} files")
    return 0


if __name__ == "__main__":
    main()