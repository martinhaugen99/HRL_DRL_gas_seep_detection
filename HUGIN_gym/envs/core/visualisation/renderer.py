from functools import lru_cache
from io import BytesIO

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy

CAMERA_FOV = 75 # vertical field of view of the Meshcat camera, in degrees
# colors (0xRRGGBB) of the gas cloud, per band of concentration_bands
CLOUD_COLORS = {"plume": 0xBDFFF9, "plume and border": 0xE08CFF, "border": 0x9933FF} # white-cyan, light and dark purple


def concentration_bands(conc, c_threshold, c_around_width):
    """
    Masks of the cells counted as plume only, as plume and border, and as border only. As in HUGIN.reset, plume cells
    are at or above c_threshold and border cells within c_around_width of it, so the two overlap.
    """
    above = conc >= c_threshold
    around = (conc >= c_threshold - c_around_width) & (conc <= c_threshold + c_around_width)
    return {"plume": above & ~around, "plume and border": above & around, "border": around & ~above}


@lru_cache
def label_png(lines: tuple[tuple[str, int], ...]) -> tuple[bytes, int, int]:
    """PNG image of text lines, each (text, color 0xRRGGBB), on a dark, half-transparent background, and its width and
    height in pixels."""
    fontsize = 14
    fig = Figure(dpi=200)
    fig.patch.set_facecolor((0, 0, 0, 0.55))
    for i, (text, color) in enumerate(lines):
        # each line 1.5 font sizes below the one before, an offset in inches from the figure, which the tight bbox moves
        transform = offset_copy(fig.transFigure, fig=fig, y=-i * 1.5 * fontsize / 72, units="inches")
        fig.text(0, 0.9, text, color=f"#{color:06x}", fontsize=fontsize, weight="bold", transform=transform)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor())
    png = buffer.getvalue()
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big") # from the PNG header
    return png, width, height


class HUGINRenderer:

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.vis = meshcat.Visualizer()
        self.vis.open()
        self.trail_markers = []  # For trajectory markers
        self.step_counter = 0    # To track how many steps have passed
        self.visited_voxel_names = set()
        self.labels = {} # the lines of the label in each corner of the view

    def create_wire_3d_grid(self, size, depth, step):

    

        half_size = size / 2
        half_depth = depth / 2

        vertices = []

        for x in np.arange(-half_size, half_size + step, step):
            for y in np.arange(-half_size, half_size + step, step):
                vertices.append([x, y, -half_depth])
                vertices.append([x, y, half_depth])

        for x in np.arange(-half_size, half_size + step, step):
            for z in np.arange(-half_depth, half_depth + step, step):
                vertices.append([x, -half_size, z])
                vertices.append([x, half_size, z])

        for y in np.arange(-half_size, half_size + step, step):
            for z in np.arange(-half_depth, half_depth + step, step):
                vertices.append([-half_size, y, z])
                vertices.append([half_size, y, z])

        vertices = np.array(vertices).T

        grid = g.LineSegments(
            g.PointsGeometry(vertices),
            g.LineBasicMaterial(color=0x222222, transparent=True,
    opacity=0.3)
        )

        self.vis["3d_grid"].set_object(grid)
        

    def create_wire_2d_grid(self, size, step):

        vertices = []
        for i in np.arange(-size/2, size/2 + step, step):
            vertices.append([i, -size/2, 0])
            vertices.append([i, size/2, 0])
            vertices.append([-size/2, i, 0])
            vertices.append([size/2, i, 0])

        vertices = np.array(vertices).T

        grid = g.LineSegments(
            g.PointsGeometry(vertices),
            g.LineBasicMaterial(color=0x222222)
        )

        self.vis["custom_grid"].set_object(grid)
        self.vis["custom_grid"].set_transform(
            tf.translation_matrix([0, 0, 0.1])
        )
        
    def render(self, model_path):
        size = 50 # world dimensions max(self.x_map_size,self.y_map_size), raise warning if self.x_map_size neq y map size
        depth = 10 # world depth , self.z_map_size
        step = 5
        if self.render_mode != "human":
            return
        
        #water_volume = g.Box([size*20, size*20, depth*20]) # the big water cube
        #water_volume_material = g.MeshPhongMaterial(
         #   color=0x1A6B9F, opacity=0.5, transparent=True
        #)
        
        #self.vis["water_volume"].set_object(water_volume, water_volume_material)
        
        print("model_path: ", model_path)
        self.vis["vessel"].set_object(
            g.DaeMeshGeometry.from_file(model_path),
            g.MeshLambertMaterial(color=0xF58427, wireframe=False),
        )

        ground = g.Box([size, size, 0.01]) # the gray thing at the bottom
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -depth/2])
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)


        # -- the grid --

        # -remove the default grid by meshcat-
        #self.vis["/Background"].set_property("visible", False) # blue background
        # Dark blue top → lighter blue bottom
        self.vis["/Background"].set_property("top_color", [0.0, 0.4, 0.8])
        self.vis["/Background"].set_property("bottom_color", [0.0, 0.2, 0.3])
        self.vis["/Grid"].set_property("visible", False)
        self.vis["/Axes"].set_property("visible", False)

        # -introduce own grid-
        
        self.create_wire_3d_grid(size,depth,step) #3D
        #self.create_wire_2d_grid(size,step) #2D

        # --initialise visited map--

    
       

    def get_robot_position(self):
        return [self.state["x"], self.state["y"], self.state["z"]]

    def plot_marker(self, position, orientation=None, marker_id=None,size=1,color=0x00FF00):
        name = f"marker_{marker_id}" if marker_id else f"marker_{len(self.trail_markers)}"

        if orientation is not None:
            # Shaft of the arrow
            shaft_length = 0.2*size
            shaft_radius = 0.1*size/2
            shaft_geom = g.Cylinder(height=shaft_length, radius=shaft_radius)
            # Rotate cylinder from Z to X
            shaft_tf = tf.rotation_matrix(np.pi / 2, [0, 1, 0])
            # Translate so base is at origin and points along +X
            

            # Head of the arrow (sphere)
            head_radius = 0.2*size/2
            head_geom = g.Sphere(head_radius)
            head_tf = tf.translation_matrix([0,shaft_length/2, 0])


            # Add shaft and head as subpaths
            self.vis[name]["shaft"].set_object(shaft_geom, g.MeshLambertMaterial(color=color))
            self.vis[name]["shaft"].set_transform(shaft_tf)

            self.vis[name]["head"].set_object(head_geom, g.MeshLambertMaterial(color=color))
            self.vis[name]["head"].set_transform(head_tf)

            # Now apply yaw rotation (around Z) to the whole arrow
            rotation = tf.rotation_matrix(orientation, [0, 0, 1])
        else:
            # No orientation → just a sphere
            self.vis[name].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=color))
            rotation = np.eye(4)

        # Final placement
        translation = tf.translation_matrix(position)
        final_tf = tf.concatenate_matrices(translation, rotation)
        self.vis[name].set_transform(final_tf)


        self.trail_markers.append(name)


    def plot_target(self, target_position):
        target_sphere = g.Sphere(0.15)
        target_material = g.MeshLambertMaterial(color=0xFF0000)  # Bright red

        self.vis["target"].set_object(target_sphere, target_material)

        transform = tf.translation_matrix(target_position)
        self.vis["target"].set_transform(transform)

    def update_visited_grid(self):
        """
        Create a green voxel only when a cell is visited.
        No pre-existing cubes are needed.
        """
        visited = self.state["visited_map"][:, :, :, 0].copy()  # shape (X,Y,Z,1)

        # Get indices of all currently visited cells
        active_indices = np.argwhere(visited == 1)

        for i, j, k in active_indices:
            name = f"cell_{i}_{j}_{k}"

            # Skip if already created
            if name in self.visited_voxel_names:
                continue
            
            self.visited_voxel_names.add(name)  # mark as created

            # Compute world coordinates
            x = i - visited.shape[0]/2 + 0.5
            y = j - visited.shape[1]/2 + 0.5
            z = k - visited.shape[2]/2 + 0.5  # raise slightly above seabed if needed

            # Create cube (slightly smaller than 1 to avoid z-fighting)
            cube = g.Box([0.9, 0.9, 0.9])
            material = g.MeshBasicMaterial(color=0x00FF00, transparent=True, opacity=0.1)

            self.vis["visited_grid"][name].set_object(cube, material)

            # Set transform
            T = tf.translation_matrix([x, y, z])
            self.vis["visited_grid"][name].set_transform(T)

    def step_sim(self, state, color=0x00FF00):
        """Move the AUV to state and add a trail marker in color (0xRRGGBB)."""
        self.state = state.copy()  # maybe wrong. check later
        if self.render_mode != "human":
            return

        translation = np.array([self.state["x"], self.state["y"], self.state["z"]])
        rotation_matrix = np.array(
            [
                [np.cos(self.state["theta"]-np.pi/2), -np.sin(self.state["theta"]-np.pi/2), 0],
                [np.sin(self.state["theta"]-np.pi/2), np.cos(self.state["theta"]-np.pi/2), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        self.vis["vessel"].set_transform(transform_matrix)
        # Plot marker every 2 steps
        self.step_counter += 1
        #if self.step_counter % 1 == 0:
        self.plot_marker(position=translation, orientation=(self.state["theta"]-np.pi/2), marker_id=self.step_counter, color=color)
        #self.update_visited_grid()

    def show_label(self, lines, corner="top_left", pixel_size=0.0011, min_aspect=1.5):
        """
        Show text lines, each (text, color 0xRRGGBB), in a corner of the view: "top_left" (e.g. the sub-agent in control)
        or "bottom_left" (the legend of the gas cloud). pixel_size: size of a pixel of the label image in the view, where
        the view is 2 * tan(CAMERA_FOV / 2) = 1.53 high.
        The label is attached to the camera, so it stays in place when the view is turned or zoomed. It is fully in view
        for browser windows at least min_aspect times as wide as high.
        """
        lines = tuple(lines)
        if corner not in ("top_left", "bottom_left"):
            raise ValueError(f"corner must be 'top_left' or 'bottom_left', not {corner!r}")
        if self.render_mode != "human" or self.labels.get(corner) == lines:
            return
        self.labels[corner] = lines
        png, width_px, height_px = label_png(lines)
        width, height = width_px * pixel_size, height_px * pixel_size
        half_view = np.tan(np.radians(CAMERA_FOV / 2)) # half the view height, 1 unit in front of the camera
        margin = 0.03
        match corner:
            case "top_left":
                y = half_view - margin - height / 2
            case "bottom_left":
                y = -half_view + margin + height / 2
        label = self.vis[f"/Cameras/default/rotated/<object>/label_{corner}"]
        label.set_object(
            g.Box([width, height, 1e-4]),
            # side=0: only the side facing the camera, the back would show the text mirrored. depthTest=False: always on top
            g.MeshBasicMaterial(map=g.ImageTexture(image=g.PngImage(png)), transparent=True, side=0, depthTest=False),
        )
        label.set_transform(tf.translation_matrix(
            [-half_view * min_aspect + margin + width / 2, y, -1.0] # the camera looks along -z
        ))

    def plot_concentration_field(self, concentration_fn,
                             gaussian_center,
                             sigma,
                             grid_min=-20, grid_max=20,
                             resolution=0.5):
        """
        Visualizes only points within radius sigma of the Gaussian center.

        Opacity:
        r = sigma   -> opacity = 0.2
        r = 0       -> opacity = 0.7
        """

        cx, cy, cz = gaussian_center

        # Remove old field
        try:
            self.vis["concentration_field"].delete()
        except KeyError:
            # Node does not exist yet, nothing to delete
            pass
    
        xs = np.arange(grid_min, grid_max, resolution)
        ys = np.arange(grid_min, grid_max, resolution)
        zs = np.arange(grid_min, grid_max, resolution)

        idx = 0

        for x in xs:
            for y in ys:
                for z in zs:

                    # Compute radial distance from Gaussian center
                    r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)

                    # Only visualize inside 1 sigma
                    if r > sigma:
                        continue

                    # Get concentration (optional, but useful for scaling)
                    c = concentration_fn(x, y, z)[0,0,0] # new definition of concentraton_fn

                    # Map opacity: r=sigma → 0.2, r=0 → 0.7
                    opacity = 0.2 + (0.7 - 0.2) * (1 - r/sigma)

                    # Sphere radius proportional to concentration
                    radius = 0.1 * float(c) # we treated the concentration before as .astype(np.float32) to work with the neural net. For the visualisation we need float ...

                    sphere = g.Sphere(radius)

                    material = g.MeshPhongMaterial(
                        color=0xBDFFF9,
                        opacity=opacity,
                        transparent=True
                    )

                    name = f"concentration_field/point_{idx}"
                    idx += 1

                    self.vis[name].set_object(sphere, material)
                    self.vis[name].set_transform(
                        tf.translation_matrix([x, y, z])
                    )

    def plot_concentration_field2(self, x,y,z,concentration_field,
                            c_threshold, c_around_width):
        """
        Draw the cells with gas as spheres, sized and shaded by concentration and colored (CLOUD_COLORS) by what the
        coverage counts them as: plume only, plume and border, or border only (concentration_bands). A legend in the
        bottom left corner of the view gives the colors and their concentrations.
        """

        # Remove old field
        try:
            self.vis["concentration_field"].delete()
        except KeyError:
            # Node does not exist yet, nothing to delete
            pass

        # --- Broadcast grid to full shape ---
        X = x + 0*y + 0*z
        Y = y + 0*x + 0*z
        Z = z + 0*x + 0*y

        # --- Normalize concentration ---
        c_max = concentration_field.max()
        Cn = concentration_field / c_max if c_max > 0 else concentration_field # all zeros edge case

        # --- Render ---
        idx = 0
        for band, mask in concentration_bands(concentration_field, c_threshold, c_around_width).items():
            for xv, yv, zv, cn in zip(X[mask], Y[mask], Z[mask], Cn[mask]):

                sphere = g.Sphere(float(0.1 + 0.2 * cn)) # radius 0.1 to 0.3, so the colors of the border cells show

                material = g.MeshPhongMaterial(
                    color=CLOUD_COLORS[band],
                    opacity=float(0.4 + (0.9 - 0.4) * cn), # low concentration → 0.4, high → 0.9
                    transparent=True
                )

                name = f"concentration_field/point_{idx}"
                idx += 1

                self.vis[name].set_object(sphere, material)
                self.vis[name].set_transform(
                    tf.translation_matrix([float(xv), float(yv), float(zv+1.0)]) # raise slightly above seabed = bring in same plane as visited map
                )

        self.show_label([
            ("Gas concentration", 0xFFFFFF),
            (f"plume: above {c_threshold + c_around_width:.2f}", CLOUD_COLORS["plume"]),
            (f"plume and border: {c_threshold:.2f} to {c_threshold + c_around_width:.2f}", CLOUD_COLORS["plume and border"]),
            (f"border: {c_threshold - c_around_width:.2f} to {c_threshold:.2f}", CLOUD_COLORS["border"]),
        ], corner="bottom_left")

    def reset(self):
        self.vis.delete()

        self.trail_markers = []
        self.step_counter = 0
        self.visited_voxel_names = set()

        
        