"""Tests for the gas cloud and the labels in the Meshcat view of HUGINRenderer, without a Meshcat server."""

import numpy as np
import pytest

from HUGIN_gym.envs.core.visualisation.renderer import (
    CAMERA_FOV,
    CLOUD_COLORS,
    HUGINRenderer,
    concentration_bands,
    label_png,
)

C_THRESHOLD = 0.3
C_AROUND_WIDTH = 0.15


class FakeVisualizer:
    """Stand-in for meshcat.Visualizer that records the commands per path."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, str, tuple]] = []

    def __getitem__(self, path: str) -> "FakeNode":
        return FakeNode(self, path)


class FakeNode:
    def __init__(self, vis: FakeVisualizer, path: str) -> None:
        self.vis = vis
        self.path = path

    def set_object(self, geometry: object, material: object) -> None:
        self.vis.commands.append((self.path, "set_object", (geometry, material)))

    def set_transform(self, matrix: np.ndarray) -> None:
        self.vis.commands.append((self.path, "set_transform", (matrix,)))

    def delete(self) -> None:
        self.vis.commands.append((self.path, "delete", ()))


def make_renderer() -> HUGINRenderer:
    renderer = HUGINRenderer.__new__(HUGINRenderer)  # without opening Meshcat
    renderer.render_mode = "human"
    renderer.labels = {}
    renderer.vis = FakeVisualizer()  # type: ignore[assignment]
    return renderer


def commands_of(renderer: HUGINRenderer) -> list[tuple[str, str, tuple]]:
    return renderer.vis.commands  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("conc", "band"),
    [
        (0.1, None),
        (0.2, "border"),
        (0.35, "plume and border"),
        (0.5, "plume"),
    ],
)
def test_concentration_bands_put_a_cell_in_one_band(
    conc: float, band: str | None
) -> None:
    # Act
    bands = concentration_bands(np.array([conc]), C_THRESHOLD, C_AROUND_WIDTH)

    # Assert
    assert [name for name, mask in bands.items() if mask[0]] == ([band] if band else [])


def test_concentration_bands_add_up_to_the_plume_and_the_border() -> None:
    # Arrange
    conc = np.random.default_rng(0).random((41, 41, 1))

    # Act
    bands = concentration_bands(conc, C_THRESHOLD, C_AROUND_WIDTH)

    # Assert: plume cells are at or above the threshold, border cells within the width of it
    plume = conc >= C_THRESHOLD
    border = (conc >= C_THRESHOLD - C_AROUND_WIDTH) & (
        conc <= C_THRESHOLD + C_AROUND_WIDTH
    )
    np.testing.assert_array_equal(bands["plume"] | bands["plume and border"], plume)
    np.testing.assert_array_equal(bands["plume and border"] | bands["border"], border)
    assert not (bands["plume"] & bands["border"]).any()


def test_label_png_is_a_png_as_large_as_its_text() -> None:
    # Act
    png, width, height = label_png((("Sub-agent: spatial exploration", 0x00FF00),))
    _, longer_width, _ = label_png(
        (("Sub-agent: gas border exploration (5 steps)", 0x00FF00),)
    )
    _, _, two_line_height = label_png(
        (("Gas concentration", 0xFFFFFF), ("plume: above 0.45", 0xBDFFF9))
    )

    # Assert
    assert png.startswith(b"\x89PNG")
    assert width > height
    assert longer_width > width
    assert two_line_height > 1.5 * height


@pytest.mark.parametrize("corner", ["top_left", "bottom_left"])
def test_show_label_is_in_its_corner_in_front_of_the_camera(corner: str) -> None:
    # Arrange
    renderer = make_renderer()
    min_aspect = 1.5
    path = f"/Cameras/default/rotated/<object>/label_{corner}"

    # Act
    renderer.show_label(
        [("Sub-agent: gas plume exploration", 0xFF3399)],
        corner=corner,
        min_aspect=min_aspect,
    )

    # Assert: inside the view of a window min_aspect times as wide as high, left of the center and in its half
    commands = commands_of(renderer)
    assert [(p, command) for p, command, _ in commands] == [
        (path, "set_object"),
        (path, "set_transform"),
    ]
    width, height, _ = commands[0][2][0].lengths
    x, y, z = commands[1][2][0][:3, 3]
    half_view = np.tan(np.radians(CAMERA_FOV / 2)) * -z
    assert z == -1.0
    assert -half_view * min_aspect < x - width / 2 < x < 0
    if corner == "top_left":
        assert 0 < y < y + height / 2 < half_view
    else:
        assert -half_view < y - height / 2 < y < 0


def test_show_label_rejects_other_corners() -> None:
    # Act and Assert
    with pytest.raises(ValueError, match="corner"):
        make_renderer().show_label([("text", 0xFFFFFF)], corner="top_right")


def test_show_label_sends_only_changed_labels() -> None:
    # Arrange
    renderer = make_renderer()
    space = [("Sub-agent: spatial exploration", 0x00FF00)]
    plume = [("Sub-agent: gas plume exploration", 0xFF3399)]

    # Act
    renderer.show_label(space)
    renderer.show_label(space)
    renderer.show_label(plume)
    renderer.show_label(plume, corner="bottom_left")

    # Assert
    commands = commands_of(renderer)
    assert sum(command == "set_object" for _, command, _ in commands) == 3
    assert renderer.labels == {"top_left": tuple(plume), "bottom_left": tuple(plume)}


def test_cloud_is_colored_by_band_with_a_legend() -> None:
    # Arrange: one cell per band and one cell without gas
    renderer = make_renderer()
    conc = np.array([[[0.1], [0.2]], [[0.35], [0.5]]])
    x = np.arange(2)[:, None, None]
    y = np.arange(2)[None, :, None]
    z = np.zeros(1)[None, None, :]

    # Act
    renderer.plot_concentration_field2(x, y, z, conc, C_THRESHOLD, C_AROUND_WIDTH)

    # Assert
    sphere_colors = [
        args[1].color
        for path, command, args in commands_of(renderer)
        if path.startswith("concentration_field/point_") and command == "set_object"
    ]
    assert sphere_colors == [
        CLOUD_COLORS["plume"],
        CLOUD_COLORS["plume and border"],
        CLOUD_COLORS["border"],
    ]
    assert renderer.labels["bottom_left"] == (
        ("Gas concentration", 0xFFFFFF),
        ("plume: above 0.45", CLOUD_COLORS["plume"]),
        ("plume and border: 0.30 to 0.45", CLOUD_COLORS["plume and border"]),
        ("border: 0.15 to 0.30", CLOUD_COLORS["border"]),
    )
