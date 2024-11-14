import colorsys
import numpy as np
import os
import xml.etree.ElementTree as ET


def cart2pol(vec):
    """Convert a cartesian 2D vector to polar coordinates."""
    r = np.sqrt(vec[0]**2 + vec[1]**2)
    theta = np.arctan2(vec[1], vec[0])
    return np.asarray([r, theta])


def get_distinct_colors(n=2):
    """Generate distinct colors for agents."""
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return RGB_tuples


def _set_class(root, prop, name):
    """Set the 'class' attribute for a specific property in the XML tree."""
    if root is None:
        return
    if root.tag == prop:
        root.set('class', name)
    for child in list(root):
        _set_class(child, prop, name)


def _add_prefix(root, prop, prefix, force_set=False):
    """Add a prefix to the property names in the XML tree."""
    if root is None:
        return
    root_prop_val = root.get(prop)
    if root_prop_val is not None:
        root.set(prop, f"{prefix}/{root_prop_val}")
    elif force_set:
        root.set(prop, f"{prefix}/anon{np.random.randint(1, 1e10)}")
    for child in list(root):
        _add_prefix(child, prop, prefix, force_set)


def _tuple_to_str(tp):
    """Convert a tuple to a space-separated string."""
    return " ".join(map(str, tp))


def construct_scene(scene_xml_path, agent_xml_paths,
                    agent_densities=None,
                    agent_scopes=None,
                    init_poses=None,
                    rgb=None,
                    tatami_size=None):
    """
    Construct an XML that represents a MuJoCo scene for sumo.

    Args:
        scene_xml_path (str): Path to the base scene XML (e.g., tatami).
        agent_xml_paths (list): List of paths to agent XML files.
        agent_densities (list): List of densities for each agent.
        agent_scopes (list): List of scopes (prefixes) for each agent.
        init_poses (list): Initial positions for each agent.
        rgb (list): List of RGB tuples for agent colors.
        tatami_size (float): Size of the tatami (sumo ring).
    """
    n_agents = len(agent_xml_paths)
    assert n_agents == 2, "Only 2-agent sumo is currently supported."

    # Parse the base scene XML
    scene = ET.parse(scene_xml_path)
    scene_root = scene.getroot()
    scene_default = scene_root.find('default')
    scene_body = scene_root.find('worldbody')
    scene_actuator = scene_root.find('actuator')
    if scene_actuator is None:
        scene_actuator = ET.SubElement(scene_root, 'actuator')
    scene_sensors = scene_root.find('sensor')
    if scene_sensors is None:
        scene_sensors = ET.SubElement(scene_root, 'sensor')

    # Set tatami size if specified
    if tatami_size is not None:
        for geom in scene_body.findall('geom'):
            name = geom.get('name')
            if name == 'tatami':
                size = tatami_size + 0.3
                geom.set('size', f"{size:.2f} {size:.2f} 0.25")
            elif name == 'topborder':
                fromto = f"-{tatami_size:.2f} {tatami_size:.2f} 0.5  {tatami_size:.2f} {tatami_size:.2f} 0.5"
                geom.set('fromto', fromto)
            elif name == 'rightborder':
                fromto = f"{tatami_size:.2f} -{tatami_size:.2f} 0.5  {tatami_size:.2f} {tatami_size:.2f} 0.5"
                geom.set('fromto', fromto)
            elif name == 'bottomborder':
                fromto = f"-{tatami_size:.2f} -{tatami_size:.2f} 0.5  {tatami_size:.2f} -{tatami_size:.2f} 0.5"
                geom.set('fromto', fromto)
            elif name == 'leftborder':
                fromto = f"-{tatami_size:.2f} -{tatami_size:.2f} 0.5  -{tatami_size:.2f} {tatami_size:.2f} 0.5"
                geom.set('fromto', fromto)

    # Resolve colors
    if rgb is None:
        rgb = list(get_distinct_colors(n_agents))
    else:
        assert len(rgb) == n_agents, "Each agent must have a color."
    RGBA_tuples = [_tuple_to_str(color + (1.0,)) for color in rgb]

    # Resolve densities
    if agent_densities is None:
        agent_densities = [10.0] * n_agents

    # Resolve scopes
    if agent_scopes is None:
        agent_scopes = [f'agent{i}' for i in range(n_agents)]
    else:
        assert len(agent_scopes) == n_agents, "Each agent must have a scope."

    # Resolve initial positions
    if init_poses is None:
        r, phi, z = 1.5, 0.0, 0.75
        delta = (2.0 * np.pi) / n_agents
        init_poses = []
        for i in range(n_agents):
            angle = phi + i * delta
            x, y = r * np.cos(angle), r * np.sin(angle)
            init_poses.append((x, y, z))

    # Build agent XMLs
    for i in range(n_agents):
        agent_xml = ET.parse(agent_xml_paths[i])

        # Create a default class for the agent
        agent_default = ET.SubElement(
            scene_default, 'default',
            attrib={'class': agent_scopes[i]}
        )

        # Set defaults
        rgba = RGBA_tuples[i]
        density = str(agent_densities[i])
        default_set = False
        agent_default_elements = agent_xml.find('default')
        if agent_default_elements is not None:
            for child in list(agent_default_elements):
                if child.tag == 'geom':
                    child.set('rgba', rgba)
                    child.set('density', density)
                    default_set = True
                agent_default.append(child)
        if not default_set:
            # If no geom defaults are set, create one
            ET.SubElement(
                agent_default, 'geom',
                attrib={
                    'density': density,
                    'contype': '1',
                    'conaffinity': '1',
                    'rgba': rgba,
                }
            )

        # Build agent body
        agent_body = agent_xml.find('worldbody/body')
        if agent_body is None:
            agent_body = agent_xml.find('body')
        if agent_body is None:
            raise ValueError("Agent XML must contain a <body> element.")

        # Set initial position
        agent_body.set('pos', _tuple_to_str(init_poses[i]))
        # Add class to all geoms
        _set_class(agent_body, 'geom', agent_scopes[i])
        # Add prefix to all names, important to map joints
        _add_prefix(agent_body, 'name', agent_scopes[i], force_set=True)
        # Add agent body to the scene
        scene_body.append(agent_body)

        # Build agent actuators
        agent_actuator = agent_xml.find('actuator')
        if agent_actuator is not None:
            # Add class and prefix to all motor joints
            _add_prefix(agent_actuator, 'joint', agent_scopes[i])
            _add_prefix(agent_actuator, 'name', agent_scopes[i])
            _set_class(agent_actuator, 'motor', agent_scopes[i])
            # Add actuators to the scene
            for motor in list(agent_actuator):
                scene_actuator.append(motor)

        # Build agent sensors
        agent_sensors = agent_xml.find('sensor')
        if agent_sensors is not None:
            # Add prefix to all sensors
            _add_prefix(agent_sensors, 'joint', agent_scopes[i])
            _add_prefix(agent_sensors, 'name', agent_scopes[i])
            # Add sensors to the scene
            for sensor in list(agent_sensors):
                scene_sensors.append(sensor)

    return scene
