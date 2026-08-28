#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import ijson


PATH = "data/WQSP_qlever/predictions/Qwen25-7b_WQSP_qlever-sparql/resolved/ChatKBQA.type_map+ChatKBQA.facc1+ChatKBQA.simple+ChatKBQA.neighborhood/WQSP_qlever_test.sparql.debug.json"

# Extracted JSON files will be placed next to this script.
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = SCRIPT_DIR


class Browser:
    def __init__(self, path):
        self.path = path

    # ------------------------------------------------------------------
    # ITERATION
    # ------------------------------------------------------------------

    def items(self):
        """
        Stream items from the top-level JSON:

        {
            "meta": {...},
            "items": [
                {...},
                {...}
            ]
        }

        Only one item is held in memory at a time.
        """
        with open(self.path, "rb") as f:
            yield from ijson.items(f, "items.item")

    def get_nth(self, n):
        """Return (index, item) for the nth item."""
        if n < 0:
            return None, None

        for i, item in enumerate(self.items()):
            if i == n:
                return i, item

        return None, None

    def get_id(self, wanted):
        """Find an item by exact ID."""
        for i, item in enumerate(self.items()):
            if item.get("id") == wanted:
                return i, item

        return None, None

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def find(self, path, value):
        """
        Exact field match.

        Example:
            find winning_pass_linker ChatKBQA.neighborhood
        """
        value = str(value)

        for i, item in enumerate(self.items()):
            actual = self.get_field(item, path)

            if str(actual) == value:
                yield i, item

    def find_contains(self, path, value):
        """
        Case-insensitive substring search.
        """
        value = str(value).lower()

        for i, item in enumerate(self.items()):
            actual = self.get_field(item, path)

            if actual is not None and value in str(actual).lower():
                yield i, item

    # ------------------------------------------------------------------
    # FIELD ACCESS
    # ------------------------------------------------------------------

    @staticmethod
    def get_field(obj, path):
        """
        Resolve a dotted path.

        Examples:

            question
            passes.0
            passes.0.pass_linker
            passes.0.beams.0.raw_beam
            passes.0.predicate_debug.0.per_label
        """

        if not path:
            return obj

        for part in path.split("."):
            if isinstance(obj, dict):
                if part not in obj:
                    return None

                obj = obj[part]

            elif isinstance(obj, list):
                try:
                    index = int(part)
                except ValueError:
                    return None

                if index < 0 or index >= len(obj):
                    return None

                obj = obj[index]

            else:
                return None

        return obj

    # ------------------------------------------------------------------
    # JSON SERIALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def json_default(value):
        """
        Handle ijson Decimal values.

        ijson commonly parses JSON numbers as Decimal.
        """
        if isinstance(value, Decimal):
            return float(value)

        return str(value)

    @classmethod
    def pretty_json(cls, value):
        return json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=cls.json_default,
        )

    # ------------------------------------------------------------------
    # PAGING
    # ------------------------------------------------------------------

    @classmethod
    def show_paged(cls, value):
        """
        Show potentially huge JSON through less.
        """

        text = cls.pretty_json(value)

        # Respect user's preferred pager if set.
        pager = os.environ.get("PAGER")

        if pager:
            command = pager.split()

        else:
            command = ["less", "-R", "-S"]

        try:
            subprocess.run(
                command,
                input=text,
                text=True,
            )

        except FileNotFoundError:
            # Fallback if less isn't installed.
            print(text)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    @staticmethod
    def summary(item, index=None):
        print()
        print("=" * 100)

        if index is not None:
            print(f"Index:               {index}")

        print(f"ID:                  {item.get('id')}")
        print(f"Question:            {item.get('question')}")
        print(f"Winning pass:        {item.get('winning_pass_index')}")
        print(f"Winning linker:      {item.get('winning_pass_linker')}")

        passes = item.get("passes", [])

        print(f"Number of passes:    {len(passes)}")

        for i, p in enumerate(passes):
            print(
                f"  Pass {i}: "
                f"index={p.get('pass_index')} "
                f"linker={p.get('pass_linker')} "
                f"found={p.get('found')} "
                f"beam_rank={p.get('beam_rank')}"
            )

        print("=" * 100)

    # ------------------------------------------------------------------
    # TREE
    # ------------------------------------------------------------------

    @classmethod
    def print_tree(cls, obj, indent=0, name=None, max_depth=None):
        """
        Print the structure of an object without dumping its contents.

        Example:

        passes
          [0]
            pass_index
            beams
              [0]
                rank
                entity_chain
                  [0]
                    linker_id
                    attempted
                    resolved
        """

        prefix = " " * indent

        if name is not None:
            if isinstance(obj, dict):
                print(f"{prefix}{name} {{")
            elif isinstance(obj, list):
                print(f"{prefix}{name} [")
            else:
                print(f"{prefix}{name}: {cls.format_scalar(obj)}")

        if max_depth is not None and indent >= max_depth:
            return

        child_indent = indent + 2

        if isinstance(obj, dict):

            for key, value in obj.items():

                if isinstance(value, dict):
                    print(f"{' ' * child_indent}{key} {{")
                    cls.print_tree(
                        value,
                        child_indent + 2,
                        max_depth=max_depth,
                    )

                elif isinstance(value, list):
                    print(
                        f"{' ' * child_indent}"
                        f"{key} [{len(value)}]"
                    )

                    # Don't expand empty lists.
                    if not value:
                        continue

                    # Expand each list element.
                    for i, element in enumerate(value):
                        element_indent = child_indent + 2

                        if isinstance(element, (dict, list)):
                            print(
                                f"{' ' * element_indent}"
                                f"[{i}]"
                            )

                            cls.print_tree(
                                element,
                                element_indent + 2,
                                max_depth=max_depth,
                            )
                        else:
                            print(
                                f"{' ' * element_indent}"
                                f"[{i}]: "
                                f"{cls.format_scalar(element)}"
                            )

                else:
                    print(
                        f"{' ' * child_indent}"
                        f"{key}: "
                        f"{cls.format_scalar(value)}"
                    )

        elif isinstance(obj, list):

            for i, element in enumerate(obj):

                element_indent = child_indent

                if isinstance(element, (dict, list)):
                    print(
                        f"{' ' * element_indent}"
                        f"[{i}]"
                    )

                    cls.print_tree(
                        element,
                        element_indent + 2,
                        max_depth=max_depth,
                    )

                else:
                    print(
                        f"{' ' * element_indent}"
                        f"[{i}]: "
                        f"{cls.format_scalar(element)}"
                    )

    @staticmethod
    def format_scalar(value):
        if isinstance(value, str):
            # Don't print enormous strings in tree mode.
            value = value.replace("\n", "\\n")

            if len(value) > 200:
                return repr(value[:197] + "...")

            return repr(value)

        if isinstance(value, Decimal):
            return str(value)

        return repr(value)

    # ------------------------------------------------------------------
    # EXTRACT
    # ------------------------------------------------------------------

    @classmethod
    def extract(cls, item, index, field=None):
        """
        Write the current item or a nested field to a JSON file next to
        this script.
        """

        if field:
            value = cls.get_field(item, field)

            if value is None:
                print(f"Field not found: {field}")
                return

            safe_name = cls.sanitize_filename(field)
            filename = (
                f"extract_{index}_{safe_name}.json"
            )

        else:
            value = item

            item_id = item.get("id", f"item_{index}")
            safe_id = cls.sanitize_filename(item_id)

            filename = f"extract_{index}_{safe_id}.json"

        output_path = EXTRACT_DIR / filename

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                value,
                f,
                indent=2,
                ensure_ascii=False,
                default=cls.json_default,
            )

            f.write("\n")

        size = output_path.stat().st_size

        print()
        print(f"Extracted to:")
        print(f"  {output_path}")
        print(f"Size: {cls.human_size(size)}")

    @staticmethod
    def sanitize_filename(value):
        result = str(value)

        for char in (
            "/",
            "\\",
            " ",
            ":",
            "*",
            "?",
            '"',
            "<",
            ">",
            "|",
        ):
            result = result.replace(char, "_")

        return result

    @staticmethod
    def human_size(size):
        units = ["B", "KB", "MB", "GB", "TB"]

        size = float(size)

        for unit in units:
            if size < 1024:
                return f"{size:.1f} {unit}"

            size /= 1024

        return f"{size:.1f} PB"


# ======================================================================
# UI
# ======================================================================


def print_help():
    print(
        """
Commands
========

Navigation
----------

  id <ID>
      Find an item by its exact ID.

      Example:
        id WebQTest-44.P0

  n <NUMBER>
      Jump to item number (0-based).

      Example:
        n 44

  next
      Go to the next item.

  prev
      Go to the previous item.


Search
------

  find <FIELD> <VALUE>
      Exact match on a field.

      Examples:
        find winning_pass_linker ChatKBQA.neighborhood
        find winning_pass_index 1
        find passes.0.pass_linker ChatKBQA.simple

  contains <FIELD> <TEXT>
      Case-insensitive substring search.

      Examples:
        contains question polk
        contains id WebQTest-42
        contains winning_pass_linker neighborhood


Inspect
-------

  show
      Show the complete current item.

      Opens through `less`, so even very large items are manageable.

  show <FIELD>
      Show a specific field.

      Examples:
        show question
        show winning_pass_linker
        show passes
        show passes.0
        show passes.0.beams
        show passes.0.predicate_debug
        show passes.0.relation_permutations_tried

  tree
      Show the complete structure of the current item without
      printing the actual large values.

  tree <FIELD>
      Show the structure below a particular field.

      Examples:
        tree
        tree passes
        tree passes.0
        tree passes.0.beams

  keys
      Show top-level fields of the current item.

  summary
      Show a compact summary of the current item.


Extract
-------

  extract
      Write the complete current item to a JSON file next to this script.

  extract <FIELD>
      Write only a nested field to a JSON file.

      Examples:
        extract
        extract passes
        extract passes.0
        extract passes.0.predicate_debug


Other
-----

  help
      Show this help.

  q
      Quit.

Search results can be selected by entering their result number.
"""
    )


def select_search_result(results):
    if not results:
        print("No matches.")
        return None

    try:
        selection = input(
            "\nSelect result number "
            "(Enter to cancel): "
        ).strip()

        if not selection:
            return None

        number = int(selection)

        if number < 0 or number >= len(results):
            print("Invalid selection.")
            return None

        return results[number]

    except ValueError:
        print("Invalid selection.")
        return None


# ======================================================================
# MAIN
# ======================================================================


def main():
    browser = Browser(PATH)

    current_index = None
    current_item = None

    print()
    print("WebQSP debug browser")
    print("====================")
    print()
    print(f"File:")
    print(f"  {PATH}")
    print()
    print("Items are streamed from disk; the 30 GB file is NOT loaded.")
    print()
    print_help()

    while True:

        try:
            command = input("\n> ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue

        # --------------------------------------------------------------
        # QUIT
        # --------------------------------------------------------------

        if command in ("q", "quit", "exit"):
            break

        # --------------------------------------------------------------
        # HELP
        # --------------------------------------------------------------

        if command in ("help", "?"):
            print_help()
            continue

        # --------------------------------------------------------------
        # ID
        # --------------------------------------------------------------

        if command.startswith("id "):

            wanted = command[3:].strip()

            print(f"Searching for {wanted!r}...")

            index, item = browser.get_id(wanted)

            if item is None:
                print("Not found.")
            else:
                current_index = index
                current_item = item

                browser.summary(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # NUMBER
        # --------------------------------------------------------------

        if command.startswith("n "):

            try:
                number = int(command[2:].strip())

            except ValueError:
                print("Usage: n <number>")
                continue

            print(f"Searching for item {number}...")

            index, item = browser.get_nth(number)

            if item is None:
                print("Not found.")
            else:
                current_index = index
                current_item = item

                browser.summary(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # FIND
        # --------------------------------------------------------------

        if command.startswith("find "):

            parts = command.split(maxsplit=2)

            if len(parts) != 3:
                print("Usage: find <field> <value>")
                continue

            field = parts[1]
            value = parts[2]

            print()
            print(
                f"Searching: "
                f"{field} == {value!r}"
            )
            print()

            results = []

            for index, item in browser.find(
                field,
                value,
            ):
                results.append((index, item))

                print(
                    f"[{len(results) - 1:3}] "
                    f"{item.get('id')} | "
                    f"{item.get('question')} | "
                    f"{item.get('winning_pass_linker')}"
                )

                # Avoid dumping absurdly many results.
                if len(results) >= 100:
                    print()
                    print("(showing first 100 matches)")
                    break

            selected = select_search_result(results)

            if selected is not None:
                current_index, current_item = selected

                browser.summary(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # CONTAINS
        # --------------------------------------------------------------

        if command.startswith("contains "):

            parts = command.split(maxsplit=2)

            if len(parts) != 3:
                print(
                    "Usage: contains <field> <text>"
                )
                continue

            field = parts[1]
            value = parts[2]

            print()
            print(
                f"Searching: "
                f"{field} contains {value!r}"
            )
            print()

            results = []

            for index, item in browser.find_contains(
                field,
                value,
            ):
                results.append((index, item))

                print(
                    f"[{len(results) - 1:3}] "
                    f"{item.get('id')} | "
                    f"{item.get('question')} | "
                    f"{item.get('winning_pass_linker')}"
                )

                if len(results) >= 100:
                    print()
                    print("(showing first 100 matches)")
                    break

            selected = select_search_result(results)

            if selected is not None:
                current_index, current_item = selected

                browser.summary(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------------

        if command == "summary":

            if current_item is None:
                print("No item selected.")

            else:
                browser.summary(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # KEYS
        # --------------------------------------------------------------

        if command == "keys":

            if current_item is None:
                print("No item selected.")

            else:
                print()

                for key in current_item:
                    print(key)

            continue

        # --------------------------------------------------------------
        # SHOW ENTIRE ITEM
        # --------------------------------------------------------------

        if command == "show":

            if current_item is None:
                print("No item selected.")

            else:
                browser.show_paged(current_item)

            continue

        # --------------------------------------------------------------
        # SHOW FIELD
        # --------------------------------------------------------------

        if command.startswith("show "):

            if current_item is None:
                print("No item selected.")
                continue

            field = command[5:].strip()

            value = browser.get_field(
                current_item,
                field,
            )

            if value is None:
                print(
                    f"Field not found: {field}"
                )

            else:
                browser.show_paged(value)

            continue

        # --------------------------------------------------------------
        # TREE ENTIRE ITEM
        # --------------------------------------------------------------

        if command == "tree":

            if current_item is None:
                print("No item selected.")

            else:
                print()
                browser.print_tree(current_item)

            continue

        # --------------------------------------------------------------
        # TREE FIELD
        # --------------------------------------------------------------

        if command.startswith("tree "):

            if current_item is None:
                print("No item selected.")
                continue

            field = command[5:].strip()

            value = browser.get_field(
                current_item,
                field,
            )

            if value is None:
                print(
                    f"Field not found: {field}"
                )

            else:
                print()
                browser.print_tree(
                    value,
                    name=field,
                )

            continue

        # --------------------------------------------------------------
        # EXTRACT ENTIRE ITEM
        # --------------------------------------------------------------

        if command == "extract":

            if current_item is None:
                print("No item selected.")

            else:
                browser.extract(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # EXTRACT FIELD
        # --------------------------------------------------------------

        if command.startswith("extract "):

            if current_item is None:
                print("No item selected.")
                continue

            field = command[8:].strip()

            value = browser.get_field(
                current_item,
                field,
            )

            if value is None:
                print(
                    f"Field not found: {field}"
                )

            else:
                browser.extract(
                    current_item,
                    current_index,
                    field,
                )

            continue

        # --------------------------------------------------------------
        # NEXT
        # --------------------------------------------------------------

        if command == "next":

            if current_index is None:
                print("No current item.")
                continue

            print(
                f"Searching for item "
                f"{current_index + 1}..."
            )

            index, item = browser.get_nth(
                current_index + 1
            )

            if item is None:
                print("No next item.")

            else:
                current_index = index
                current_item = item

                browser.summary(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # PREVIOUS
        # --------------------------------------------------------------

        if command == "prev":

            if current_index is None:
                print("No current item.")
                continue

            if current_index == 0:
                print("Already at first item.")
                continue

            print(
                f"Searching for item "
                f"{current_index - 1}..."
            )

            index, item = browser.get_nth(
                current_index - 1
            )

            if item is None:
                print("No previous item.")

            else:
                current_index = index
                current_item = item

                browser.summary(
                    current_item,
                    current_index,
                )

            continue

        # --------------------------------------------------------------
        # UNKNOWN
        # --------------------------------------------------------------

        print(
            "Unknown command. "
            "Type 'help' for commands."
        )


if __name__ == "__main__":
    main()