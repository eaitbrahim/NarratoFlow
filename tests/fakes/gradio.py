"""A stand-in for the `gradio` module: records components and event wiring, runs handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FakeError(Exception):
    """`gr.Error`."""


class Component:
    kind = "component"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.label: str = (
            kwargs.get("label") or kwargs.get("elem_id") or (str(args[0]) if args else "")
        )

    def __repr__(self) -> str:
        return f"{self.kind}({self.label!r})"


class Markdown(Component):
    kind = "markdown"


class Dropdown(Component):
    kind = "dropdown"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.changed: list[Wiring] = []

    def change(self, fn: Callable[..., Any], inputs: Any = None, outputs: Any = None) -> None:
        self.changed.append(Wiring(fn, list(inputs or []), list(outputs or [])))


class Textbox(Component):
    kind = "textbox"


class Number(Component):
    kind = "number"


class Audio(Component):
    kind = "audio"


class Image(Component):
    kind = "image"


class File(Component):
    kind = "file"


class Checkbox(Component):
    kind = "checkbox"


class Button(Component):
    kind = "button"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.label = str(args[0]) if args else str(kwargs.get("value", ""))
        self.wired: list[Wiring] = []

    def click(self, fn: Callable[..., Any], inputs: Any = None, outputs: Any = None) -> None:
        self.wired.append(Wiring(fn, list(inputs or []), list(outputs or [])))


class Wiring:
    def __init__(self, fn: Callable[..., Any], inputs: list[Any], outputs: list[Any]) -> None:
        self.fn, self.inputs, self.outputs = fn, inputs, outputs

    def call(self, *values: Any) -> Any:
        return self.fn(*values)


class _Context:
    def __init__(self, gr: FakeGradio) -> None:
        self.gr = gr

    def __enter__(self) -> _Context:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class Blocks(_Context):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.loads: list[Wiring] = []
        self.launched: dict[str, Any] | None = None

    def load(self, fn: Callable[..., Any], inputs: Any = None, outputs: Any = None) -> None:
        self.loads.append(Wiring(fn, list(inputs or []), list(outputs or [])))

    def launch(self, **kwargs: Any) -> None:
        self.launched = kwargs


class FakeGradio:
    """Every component the station uses, remembered in creation order in `components`."""

    Error = FakeError

    def __init__(self) -> None:
        self.components: list[Component] = []
        self.blocks: Blocks | None = None

    def _make(self, cls: type[Component]) -> Callable[..., Component]:
        def build(*args: Any, **kwargs: Any) -> Component:
            made = cls(*args, **kwargs)
            self.components.append(made)
            return made

        return build

    def Blocks(self, *args: Any, **kwargs: Any) -> Blocks:
        self.blocks = Blocks(*args, **kwargs)
        return self.blocks

    def Row(self, *args: Any, **kwargs: Any) -> _Context:
        return _Context(self)

    def __getattr__(self, name: str) -> Callable[..., Component]:
        every: tuple[type[Component], ...] = (
            Markdown, Dropdown, Textbox, Number, Audio, Image, File, Checkbox, Button,
        )  # fmt: skip
        classes = {c.__name__: c for c in every}
        if name not in classes:
            raise AttributeError(name)
        return self._make(classes[name])

    def buttons(self) -> dict[str, Button]:
        return {c.label: c for c in self.components if isinstance(c, Button)}
