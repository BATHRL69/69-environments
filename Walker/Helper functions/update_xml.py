import os
import gymnasium as gym


def update_env_xml(add_obstacles, add_slippy_surface, reduce_motor_strength):
    default_path = os.path.join(
        os.path.dirname(gym.__file__), "envs", "mujoco", "assets", "ant.xml"
    )

    with open(default_path, "r") as f:
        xml_string = f.read()

    block_obstacle_xml = """
        <!-- Front wall -->
        <body name="wall_front" pos="0 2 0.5">
            <geom name="wf" type="box" size="2.5 0.1 0.8" rgba="0.8 0.8 0.8 0.8"/>
        </body>
        
        <!-- Back wall -->
        <body name="wall_back" pos="0 -2 0.5">
            <geom name="wb" type="box" size="2.5 0.1 0.8" rgba="0.8 0.8 0.8 0.8"/>
        </body>
        
        <!-- Left wall -->
        <body name="wall_left" pos="-2 0 0.5">
            <geom name="wl" type="box" size="0.1 2.5 0.8" rgba="0.8 0.8 0.8 0.8"/>
        </body>
        
        <!-- Right wall -->
        <body name="wall_right" pos="2 0 0.5">
            <geom name="wr" type="box" size="0.1 2.5 0.8" rgba="0.8 0.8 0.8 0.8"/>
        </body>
    """

    wall_obstacle_xml = """
        <!-- Front wall -->
        <body name="wall_front_1" pos="-2 2 0.5">
            <geom name="wf1" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_front_2" pos="0 2 0.5">
            <geom name="wf2" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_front_3" pos="2 2 0.5">
            <geom name="wf3" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        
        <!-- Back wall -->
        <body name="wall_back_1" pos="-2 -2 0.5">
            <geom name="wb1" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_back_2" pos="0 -2 0.5">
            <geom name="wb2" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_back_3" pos="2 -2 0.5">
            <geom name="wb3" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        
        <!-- Left wall -->
        <body name="wall_left_1" pos="-2 -2 0.5">
            <geom name="wl1" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_left_2" pos="-2 0 0.5">
            <geom name="wl2" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_left_3" pos="-2 2 0.5">
            <geom name="wl3" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        
        <!-- Right wall -->
        <body name="wall_right_1" pos="2 -2 0.5">
            <geom name="wr1" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_right_2" pos="2 0 0.5">
            <geom name="wr2" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
        <body name="wall_right_3" pos="2 2 0.5">
            <geom name="wr3" type="cylinder" size="0.2 0.8" rgba="0.8 0.8 0.8 1"/>
        </body>
    """
    # Add my obstacles
    obstacle_xml = """
       <body name="obstacle_1" pos="2 2 0.5">
            <geom name="obs1" type="cylinder" size="0.2 0.5" rgba="1 0 0 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_2" pos="4 4 0.5">
            <geom name="obs2" type="cylinder" size="0.2 0.5" rgba="1 0 0 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_3" pos="6 6 0.5">
            <geom name="obs3" type="cylinder" size="0.2 0.5" rgba="1 0 0 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_4" pos="3 -2 0.5">
            <geom name="obs4" type="cylinder" size="0.3 0.7" rgba="0 1 0 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_5" pos="-2 3 0.5">
            <geom name="obs5" type="cylinder" size="0.15 0.4" rgba="0 0 1 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_6" pos="5 0 0.5">
            <geom name="obs6" type="cylinder" size="0.25 0.6" rgba="1 1 0 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_7" pos="-3 -3 0.5">
            <geom name="obs7" type="cylinder" size="0.2 0.8" rgba="1 0 1 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_8" pos="0 5 0.5">
            <geom name="obs8" type="cylinder" size="0.4 0.3" rgba="0 1 1 1" contype="1" conaffinity="1"/>
        </body>
        <body name="obstacle_9" pos="8 -1 0.5">
            <geom name="obs9" type="cylinder" size="0.3 0.5" rgba="0.5 0.5 0 1" contype="1" conaffinity="1"/>
        </body>
    """
    if add_obstacles:
        modified_xml = xml_string.replace(
            "</worldbody>", f"{obstacle_xml}\n</worldbody>"
        )
    if add_slippy_surface:
        pass

    if reduce_motor_strength:
        pass
    custom_xml_path = (
        "C:/Users/fabie/OneDrive - University of Bath/Desktop/RL69/69-environments/Walker/Testing functions/temp_ant.xml"  # UPDATE ME TO YOUR path
    )
    with open(custom_xml_path, "w") as f:
        f.write(modified_xml)
    return str(custom_xml_path)
