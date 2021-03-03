"""Convert a Joplin raw dir export to a Notable-compatible directory structure"""

import argparse
import enum
import os
import re
import shutil
from abc import abstractmethod, ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from random import choices
from string import ascii_letters, digits
from typing import (
    Dict,
    Optional,
    Match,
    MutableMapping,
    Iterator,
    Set,
    ClassVar,
    Type,
    Any,
    List,
    Callable,
)


__author__ = "abk16"
__version__ = "0.0.1-alpha"


NOTABLE_NOTES_SUBDIR: str = "notes"
NOTABLE_ATTACHMENTS_SUBDIR: str = "attachments"
JOPLIN_RESOURCES_SUBDIR: str = "resources"


def parse_dt(jsdate: str) -> datetime:
    """Parse a js-like datetime string into a :class:`datetime.datetime` object."""
    return datetime.fromisoformat(re.sub(r"(.*)T(.*)Z", r"\1 \2", jsdate)).replace(
        tzinfo=timezone.utc
    )


def dump_dt(dt: datetime) -> str:
    """Create a js-like datetime string from a :class:`datetime.datetime` object."""
    return (
        dt.astimezone(tz=timezone.utc)
        .replace(tzinfo=None)
        .strftime("%Y-%m-%dT%H:%M:S.%fZ")
    )


class NodeType(enum.Enum):
    """
    Enum for Joplin node types.
    (N.B. only the reverse-engineered ones are defined here, and names are non-official)
    """

    NOTE = 1
    NOTEBOOK = 2
    ATTACHMENT = 4
    TAG = 5
    TAGREF = 6


@dataclass
class ParserContext:
    """
    Dataclass to hold Joplin node parser context data.
    """

    id: str
    file_path: str
    wholefile: str
    title: Optional[str]
    contents: Optional[str]
    metadata: MutableMapping[str, str]


@dataclass
class JoplinNode(ABC):
    """
    Base abstract dataclass for Joplin nodes (`.md` files).
    """

    path: str
    id: str
    parent_id: Optional[str]
    user_created_time: datetime
    user_updated_time: datetime

    _type_classes: ClassVar[Dict[NodeType, Type["JoplinNode"]]] = {}
    cacheable: ClassVar[bool] = True

    @classmethod
    @abstractmethod
    def type(cls) -> NodeType:
        """
        Abstract class method that concrete subclasses must implement to specify
        what kind of Joplin's node type they represent.
        """

    def __init_subclass__(cls, **kwargs):
        """
        Initializes subclasses, registering their Joplin node type in the base class'
        `_type_classes` mapping.

        :raise NameError: if multiple subclasses exist for the same node type.
        """
        type_: NodeType = cls.type()
        if isinstance(type_, NodeType):
            if type_ in cls._type_classes:
                raise NameError(f"Multiple classes for same node type {type_}")
            cls._type_classes[type_] = cls

    @classmethod
    def for_type(cls, type_: NodeType) -> Type["JoplinNode"]:
        """Get the appropriate :class:`JoplinNode` subclass for the given node type."""
        type_cls: Type["JoplinNode"] = cls._type_classes[type_]
        return type_cls

    @classmethod
    def from_parser(cls, context: ParserContext, **kwargs: Any) -> "JoplinNode":
        """
        Create a new :class:`JoplinNode` instance using the data from the node parser.

        :param context: the :class:`ParserContext` object with the Joplin node parser
            context data.
        :param kwargs: additional keyword-only arguments to initialize the node instance.
            Use this from subclasses to pass their specific fields' values.
        :return: the newly-created :class:`JoplinNode` instance.
        """
        return cls(
            path=context.file_path,
            id=context.id,
            parent_id=context.metadata.get("parent_id") or None,
            user_created_time=parse_dt(context.metadata["user_created_time"]),
            user_updated_time=parse_dt(context.metadata["user_updated_time"]),
            **kwargs,
        )


@dataclass
class JoplinNodeWithTitle(JoplinNode, ABC):
    """
    Mixin abstract dataclass for a Joplin node with title metadata.
    """

    title: str

    @classmethod
    def from_parser(cls, context: ParserContext, **kwargs: Any) -> "JoplinNode":
        if context.title is None:
            raise ParsingError(f'Got no title for file "{context.file_path}"')
        return super().from_parser(context, title=context.title, **kwargs)


@dataclass
class JoplinNote(JoplinNodeWithTitle):
    """
    Dataclass for Joplin "note" node types.
    """

    contents: Optional[str]

    cacheable: ClassVar[bool] = False

    @classmethod
    def type(cls) -> NodeType:
        return NodeType.NOTE

    @classmethod
    def from_parser(cls, context: ParserContext, **kwargs: Any) -> "JoplinNode":
        return super().from_parser(context, contents=(context.contents or ""), **kwargs)


@dataclass
class JoplinNotebook(JoplinNodeWithTitle):
    """
    Dataclass for Joplin "notebook" node types.
    """

    @classmethod
    def type(cls) -> NodeType:
        return NodeType.NOTEBOOK


@dataclass
class JoplinAttachment(JoplinNode):
    """
    Dataclass for Joplin "attachment" node types.
    """

    filename: str
    file_extension: str
    # TODO: support for more metadata / encryption?

    @classmethod
    def type(cls) -> NodeType:
        return NodeType.ATTACHMENT

    @classmethod
    def from_parser(cls, context: ParserContext, **kwargs: Any) -> "JoplinNode":
        return super().from_parser(
            context,
            filename=(context.metadata["filename"] or context.title),
            file_extension=context.metadata["file_extension"],
            **kwargs,
        )


@dataclass
class JoplinTag(JoplinNodeWithTitle):
    """
    Dataclass for Joplin "tag" node types.
    """

    @classmethod
    def type(cls) -> NodeType:
        return NodeType.TAG


@dataclass
class JoplinTagRef(JoplinNode):
    """
    Dataclass for Joplin "tagref" node types.
    """

    note_id: str
    tag_id: str

    @classmethod
    def type(cls) -> NodeType:
        return NodeType.TAGREF

    @classmethod
    def from_parser(cls, context: ParserContext, **kwargs: Any) -> "JoplinNode":
        return super().from_parser(
            context,
            note_id=context.metadata["note_id"],
            tag_id=context.metadata["tag_id"],
            **kwargs,
        )


class ParsingError(Exception):
    """Exception class for Joplin nodes parsing errors."""


class JoplinNodes(MutableMapping[str, JoplinNode]):
    """
    Class for mutable mapping interfaces to a Joplin raw export directory.
    It can be used as a dict, with node ids as keys, and node instances as values.
    The node instances are cached once loaded for the first time, unless the class
    for their specific node type is not `.cacheable`.
    """

    def __init__(self, base_dir: str):
        """
        Initialization for :class:`JoplinNodes`.

        :param base_dir: the base directory of the Joplin raw export.
        """
        self._base_dir: str = base_dir
        self._known_ids: Set[str] = set(self._find_ids())
        self._nodes_cache: Dict[str, JoplinNode] = {}

    @property
    def base_dir(self) -> str:
        """The Joplin raw export base directory."""
        return self._base_dir

    def _find_ids(self) -> Iterator[str]:
        """Internal method that lists all the possible node files in the base directory."""
        for filename in os.listdir(self.base_dir):
            # TODO: Only matching regex? (32-chars hex ids)
            if not filename.endswith(".md"):
                continue
            yield os.path.splitext(filename)[0]

    def _md_parse(self, id_: str) -> JoplinNode:
        """
        Internal method that parses a node file from the base directory.

        :param id_: the id of the node.
        :raise ParsingError: for any blocking parsing error.
        :return: the parsed :class:`JoplinNode` object.
        """
        newline: str = r"\n"

        file_path: str = os.path.join(self.base_dir, id_ + ".md")

        with open(file_path, "r") as fp:
            wholefile: str = fp.read()
        if not wholefile.strip():
            raise ParsingError(f'Unparseable file "{file_path}"')

        title: Optional[str] = None
        title_match: Optional[Match[str]] = re.match(
            r"^([^\r\n]*)[\r\n]{2,}", wholefile
        )
        if title_match is not None:
            title = title_match.group(1).strip()

        meta_line_pat: str = rf"(?P<name>\w+): +(?P<value>[^{newline}]*)"
        meta_match: Optional[Match[str]]
        meta_match = re.search(rf"(?:{meta_line_pat}[{newline}]?)+$", wholefile)
        if meta_match is None:
            raise ParsingError(f'Couldn\'t parse metadata for file "{file_path}"')
        meta_lines: str = meta_match.group(0).strip() + "\n"
        metadata: Dict[str, str] = {
            match.group("name"): match.group("value")
            for match in re.finditer(rf"^{meta_line_pat}$", meta_lines, flags=re.M)
        }

        try:
            type_val: int = int(metadata["type_"])
            type_: NodeType = NodeType(type_val)
        except (KeyError, ValueError) as exc:
            raise ParsingError(f'Couldn\'t parse type for file "{file_path}"') from exc

        contents: Optional[str] = None
        if title_match and meta_match:
            contents = wholefile[title_match.end() : meta_match.start()].strip()
            if contents:
                contents += "\n"

        context: ParserContext = ParserContext(
            id=id_,
            file_path=file_path,
            wholefile=wholefile,
            title=title,
            contents=contents,
            metadata=metadata,
        )

        node: JoplinNode = JoplinNode.for_type(type_).from_parser(context)
        if node.cacheable:
            self[id_] = node
        return node

    def __setitem__(self, key: str, value: JoplinNode) -> None:
        self._nodes_cache[key] = value

    def __delitem__(self, key: str) -> None:
        del self._nodes_cache[key]

    def __getitem__(self, key: str) -> JoplinNode:
        if key not in self._known_ids:
            raise KeyError(f"Node with id {key} not found")
        if key in self._nodes_cache:
            return self._nodes_cache[key]
        try:
            return self._md_parse(key)
        except ParsingError as exc:
            # TODO: Add to some blacklist of failed files/ids?
            raise KeyError from exc

    def __len__(self) -> int:
        return len(self._known_ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self._known_ids)


def find_nonconflicting_filename(filename_maker: Callable[[str], str]) -> str:
    """
    Finds a filename that doesn't conflict with existing files by adding a
    random 4-characters string in the name.

    :param filename_maker: a callable function that must accept the random name part,
        which is an empty "" string on first try, and builds a complete file path
        for validation.
    :return: the first valid non-existing file path found.
    """
    random_part: str = ""
    while True:
        file_path: str = filename_maker(random_part)
        if not os.path.exists(file_path):
            break
        random_part = "-" + "".join(choices(ascii_letters + digits, k=4))
    return file_path


def sanitize(name: str, placeholder: str = "_") -> str:
    """
    Sanitize a string for filename usage.

    :param name: the string to sanitize.
    :param placeholder: the placeholder for the bad characters. Defaults to "_".
    :return: the sanitized string.
    """
    safe_map: Dict[int, str] = {ord(c): placeholder for c in r'\/*?:"<>|'}
    return name.translate(safe_map)


def notable_md_export(
    base_dir: str, joplin_note: JoplinNote, nodes: JoplinNodes
) -> None:
    """
    Export a Joplin note node to the specified Notable storage base directory.

    :param base_dir: the base directory where Notable notes are (or will be) stored.
    :param joplin_note: the :class:`JoplinNote` object to export
    :param nodes: a :class:`JoplinNodes` instance from which the ``joplin_note``
        object originates, needed to cross-reference and export other metadata,
        such as tags, notebooks, attachments, etc.
    """
    os.makedirs(os.path.join(base_dir, NOTABLE_NOTES_SUBDIR), exist_ok=True)
    safe_title: str = sanitize(joplin_note.title)
    file_path: str = find_nonconflicting_filename(
        lambda random_part: os.path.join(
            base_dir, NOTABLE_NOTES_SUBDIR, safe_title + random_part + ".md"
        )
    )

    def mk_notebooks_tag(
        parent_id: str, seen_set: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Return a list of notebooks tags for the current note.
        They will be searched recursively and returned in a list ordered from
        the farthest to the closest ancestor (sub-notebooks are supported in Joplin).
        """
        nonlocal joplin_note, nodes
        if seen_set is None:
            seen_set = set()
        if parent_id in seen_set:
            raise ValueError(f"Got nodes recursion exporting node {joplin_note.id}")
        seen_set.add(parent_id)
        node: Optional[JoplinNode] = nodes.get(parent_id)
        if not isinstance(node, JoplinNotebook):
            raise ValueError(f"Got non-notebook ancestor for node {joplin_note.id}")
        parents: List[str] = []
        if node.parent_id is not None:
            parents = mk_notebooks_tag(node.parent_id)
        return [*parents, node.title]

    def mk_tags() -> List[str]:
        """Return a list of tags for the current note."""
        nonlocal joplin_note, nodes
        return [
            nodes[tagref.tag_id].title
            for tagref in nodes.values()
            if isinstance(tagref, JoplinTagRef) and tagref.note_id == joplin_note.id
        ]

    def attachment_replace(attach_match: Match) -> str:
        """
        Function compatible with `re.sub()`'s `repl` parameter that receives a
        regex match for a possible attachment reference candidate, checks if it's
        a valid Joplin attachment node, copies it to Notable's attachments dir,
        and returns the replaced reference for the Notable note content.
        """
        nonlocal nodes
        wholematch: str = attach_match.group(0)
        attach_id: str = attach_match.group(1)
        node: Optional[JoplinNode] = nodes.get(attach_id)
        if not isinstance(node, JoplinAttachment):
            return wholematch
        joplin_attach_path: str = os.path.join(
            nodes.base_dir,
            JOPLIN_RESOURCES_SUBDIR,
            attach_id + "." + node.file_extension,
        )
        if not os.path.exists(joplin_attach_path):
            return wholematch
        os.makedirs(os.path.join(base_dir, NOTABLE_ATTACHMENTS_SUBDIR), exist_ok=True)
        attach_basename: str = os.path.splitext(node.filename)[0].replace(" ", "_")
        notable_attach_path: str = find_nonconflicting_filename(
            lambda random_part: os.path.join(
                base_dir,
                NOTABLE_ATTACHMENTS_SUBDIR,
                (attach_basename + random_part + "." + node.file_extension),
            )
        )
        shutil.copy(joplin_attach_path, notable_attach_path)
        attach_mtime: float = node.user_updated_time.timestamp()
        os.utime(joplin_attach_path, (attach_mtime, attach_mtime))
        return f"@attachment/{os.path.basename(notable_attach_path)}"

    # Replace attachments references, and export them while we're at it
    # FIXME: take into account multiple references for the same attachment id somehow
    note_contents: str = re.sub(
        r":/([a-f0-9]{32})\b", attachment_replace, joplin_note.contents, flags=re.I
    )

    tags: List[str] = []
    if joplin_note.parent_id is not None:
        tags.append("Notebooks/" + "/".join(mk_notebooks_tag(joplin_note.parent_id)))
    tags.extend(mk_tags())

    metadata_line: str = "{name}: {value}\n"
    with open(file_path, "w") as fp:
        fp.write("---\n")
        fp.write(metadata_line.format(name="tags", value=f"[{', '.join(tags)}]"))
        fp.write(metadata_line.format(name="title", value=joplin_note.title))
        fp.write(
            metadata_line.format(
                name="created", value=dump_dt(joplin_note.user_created_time)
            )
        )
        fp.write(
            metadata_line.format(
                name="modified", value=dump_dt(joplin_note.user_updated_time)
            )
        )
        fp.write("---\n\n")
        fp.write(note_contents)


def joplin_dir_to_notable(joplin_dir: str, notable_dir: str) -> None:
    """
    Convert a Joplin raw directory export to a Notable-compatible files/directory.

    :param joplin_dir: the base directory of the Joplin raw export.
    :param notable_dir: the base directory where Notable notes are (or will be) stored.
    """
    nodes: JoplinNodes = JoplinNodes(joplin_dir)

    for node in nodes.values():
        if isinstance(node, JoplinNote):
            notable_md_export(notable_dir, node, nodes=nodes)


def main():
    """Main program entry point"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="joplin_to_notable.py", description=__doc__
    )
    parser.add_argument(
        "joplin_export",
        help="The directory path where the Joplin raw export is located",
    )
    parser.add_argument(
        "notable_dest",
        help="The destination directory where to create the notes for Notable",
    )

    args: argparse.Namespace = parser.parse_args()

    joplin_dir_to_notable(args.joplin_export, args.notable_dest)


if __name__ == "__main__":
    main()
