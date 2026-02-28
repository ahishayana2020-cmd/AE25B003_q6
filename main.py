from robot import Robot
import os
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import math
import csv
import numpy as np

# -----------------------------------------------------
# Runtime modes & visualization toggles
# -----------------------------------------------------
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True

# Part 4: Obstacle avoidance mode
AVOID_OBSTACLES = True  # Set to True to enable obstacle avoidance

# -----------------------------------------------------
# Control commands
# -----------------------------------------------------
v = 0.0
w = 0.0

# Previous mode tracking for safe transitions
prev_mode = "MANUAL"

# -----------------------------------------------------
# Load path globally
# -----------------------------------------------------
def load_path(filename):
    path_data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                path_data.append([float(row[0]), float(row[1])])
            except (ValueError, IndexError):
                # Skip headers or malformed rows
                continue
    return np.array(path_data)

path = load_path(os.path.join(os.path.dirname(__file__), "path.csv"))

# -----------------------------------------------------
# PART 2: PURE PURSUIT CONTROLLER
# -----------------------------------------------------
def pure_pursuit(x, y, theta, path, lookahead_dist=1.5, base_speed=2.5):
    
    dists = np.linalg.norm(path - np.array([x, y]), axis=1) 
    nearest_idx = np.argmin(dists)

    # Find lookahead point
    lookahead_idx = nearest_idx
    while (lookahead_idx < len(path) - 1 and
           np.linalg.norm(path[lookahead_idx] - np.array([x, y])) < lookahead_dist):
        lookahead_idx += 1

    target = path[lookahead_idx]

    # Transform target to robot frame
    dx = target[0] - x
    dy = target[1] - y

    # Robot frame coordinates
    x_r = math.cos(theta) * dx + math.sin(theta) * dy
    y_r = -math.sin(theta) * dx + math.cos(theta) * dy

    L = math.sqrt(x_r**2 + y_r**2)

    if L < 1e-6:
        return 0.0, 0.0

    # Pure pursuit law: curvature = 2 * y_r / L^2
    kappa = 2 * y_r / (L**2)

    # Speed reduction based on curvature (slow down on sharp turns)
    v = base_speed * math.exp(-abs(kappa))
    w = v * kappa

    # Check if near end of path
    if nearest_idx >= len(path) - 5:
        return 0.0, 0.0

    return v, w


# -----------------------------------------------------
# PART 3: OBSTACLE DETECTION & COLLISION AVOIDANCE
# -----------------------------------------------------
def obstacle_in_front(lidar_ranges, lidar_points,
                      safe_dist=0.8,
                      width=0.7):
    
    for r, point in zip(lidar_ranges, lidar_points):

        # Ignore max-range readings (no obstacle)
        if r >= 4.0:   # 4.0 = max_range used in LidarScan
            continue

        x, y = point

        # In front of robot
        if x > 0:

            # Within forward corridor
            if abs(y) < width:

                if r < safe_dist:
                    return True

    return False


# -----------------------------------------------------
# NEW: EMERGENCY BRAKING CHECK
# -----------------------------------------------------
def obstacle_critically_close(lidar_ranges, lidar_points,
                               critical_dist=0.4,
                               width=0.7):
    
    for r, point in zip(lidar_ranges, lidar_points):
        if r >= 4.0:
            continue
            
        x, y = point
        
        if x > 0 and abs(y) < width and r < critical_dist:
            return True
    
    return False


# -----------------------------------------------------
# PART 4: OBSTACLE AVOIDANCE STRATEGY
# -----------------------------------------------------
def get_avoidance_command(lidar_ranges, lidar_points, goal_direction):

    num_rays = len(lidar_ranges)
    half = num_rays // 2
    
    # Split into left and right sectors
    left_ranges = lidar_ranges[:half]
    right_ranges = lidar_ranges[half:]
    
    # Calculate average clearance on each side
    left_clearance = np.mean([r for r in left_ranges if r < 3.5])
    right_clearance = np.mean([r for r in right_ranges if r < 3.5])
    
    # If both sides are blocked, stop
    if left_clearance < 0.5 and right_clearance < 0.5:
        return 0.0, 0.0
    
    # Steer toward freer side
    if left_clearance > right_clearance:
        # Turn left
        return 0.5, 1.5
    else:
        # Turn right
        return 0.5, -1.5


# -----------------------------------------------------
# KEYBOARD CONTROL
# -----------------------------------------------------
def on_key(event):
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM, AVOID_OBSTACLES, prev_mode

    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        return

    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        return

    if event.key == 'm':
        prev_mode = MODE
        MODE = "MANUAL"
        v = 0.0
        w = 0.0
        print("MANUAL mode")
        return

    if event.key == 'a':
        prev_mode = MODE
        MODE = "AUTO"
        # Note: v and w will be handled in main loop with emergency braking check
        print("AUTO mode (with obstacle avoidance)" if AVOID_OBSTACLES else "AUTO mode (path following only)")
        return
    
    if event.key == 'v':
        AVOID_OBSTACLES = not AVOID_OBSTACLES
        print(f"Obstacle avoidance: {'ENABLED' if AVOID_OBSTACLES else 'DISABLED'}")
        return

    if MODE != "MANUAL":
        return

    if event.key == 'up':
        v += 1.5
    elif event.key == 'down':
        v -= 1.5
    elif event.key == 'left':
        w += 2.0
    elif event.key == 'right':
        w -= 2.0
    elif event.key == ' ':
        v = 0.0
        w = 0.0

    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 6.0), -6.0)


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    robot = Robot()

    plt.close('all')
    fig = plt.figure(num=2, figsize=(10, 10))
    fig.canvas.manager.set_window_title("Question 6 - Complete Solution (FIXED)")
    fig.canvas.mpl_connect("key_press_event", on_key)
    
    # Add instructions
    print("="*60)
    print("QUESTION 6 - COMPLETE SOLUTION (FIXED)")
    print("="*60)
    print("Controls:")
    print("  'm' - Manual mode")
    print("  'a' - Auto mode (path following + obstacle avoidance)")
    print("  'v' - Toggle obstacle avoidance on/off")
    print("  'l' - Toggle LiDAR visualization")
    print("  'o' - Toggle odometry visualization")
    print("  Arrow keys - Manual control (in manual mode)")
    print("  Space - Stop")
    print("="*60)
    print("Part 1: Odometry bug FIXED (removed conversion_factor)")
    print("Part 2: Pure Pursuit path following IMPLEMENTED")
    print("Part 3: Collision detection IMPLEMENTED")
    print("Part 4: Obstacle avoidance IMPLEMENTED")
    print("FIX: Emergency braking on mode switch near obstacles")
    print("="*60)
    
    plt.show(block=False)

    dt = 0.01
    
    # State for obstacle avoidance
    avoiding_obstacle = False
    
    # Track mode transitions
    mode_just_switched = False

    while plt.fignum_exists(fig.number):

        # Ground truth (for LiDAR)
        real_x, real_y, real_theta = robot.get_ground_truth()

        # Odometry (for control)
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()

        # LiDAR scan
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = \
            lidar.get_scan((real_x, real_y, real_theta))

        # -------------------------------------------------
        # DETECT MODE TRANSITIONS
        # -------------------------------------------------
        if MODE != prev_mode:
            mode_just_switched = True
            prev_mode = MODE
        else:
            mode_just_switched = False

        # -------------------------------------------------
        # AUTO MODE - ALL PARTS INTEGRATED WITH FIXES
        # -------------------------------------------------
        if MODE == "AUTO":
            
            # CRITICAL FIX: Check for emergency braking on mode switch
            critically_close = obstacle_critically_close(lidar_ranges, lidar_points)
            
            # If just switched to AUTO and obstacle is critically close, EMERGENCY STOP
            if mode_just_switched and critically_close:
                print("EMERGENCY BRAKE: Obstacle critically close on mode switch!")
                v = 0.0
                w = 0.0
                avoiding_obstacle = False
            else:
                # Normal obstacle detection
                obstacle_detected = obstacle_in_front(lidar_ranges, lidar_points)

                if obstacle_detected:
                    if AVOID_OBSTACLES:
                        # PART 4: Obstacle avoidance
                        # Calculate goal direction
                        dists = np.linalg.norm(path - np.array([ideal_x, ideal_y]), axis=1)
                        nearest_idx = np.argmin(dists)
                        if nearest_idx < len(path) - 1:
                            goal_point = path[nearest_idx + 10]  # Look ahead
                            goal_dir = math.atan2(goal_point[1] - ideal_y, 
                                                  goal_point[0] - ideal_x)
                            v, w = get_avoidance_command(lidar_ranges, lidar_points, goal_dir)
                            avoiding_obstacle = True
                        else:
                            v, w = 0.0, 0.0
                    else:
                        # Just stop (Part 3 only)
                        v = 0.0
                        w = 0.0
                        avoiding_obstacle = False
                else:
                    # PART 2: Pure Pursuit path following
                    v, w = pure_pursuit(
                        ideal_x,
                        ideal_y,
                        ideal_theta,
                        path,
                        lookahead_dist=1.5,
                        base_speed=2.5
                    )
                    avoiding_obstacle = False

        # -------------------------------------------------
        # Step robot
        # -------------------------------------------------
        robot.step(
            lidar_points,
            lidar_rays,
            lidar_hits,
            v,
            w,
            dt,
            show_lidar=SHOW_LIDAR,
            show_odom=SHOW_ODOM
        )
        
        # Update title with status
        status = f"Mode: {MODE}"
        if MODE == "AUTO":
            # Check current state for display
            currently_critical = obstacle_critically_close(lidar_ranges, lidar_points)
            currently_detected = obstacle_in_front(lidar_ranges, lidar_points)
            
            if mode_just_switched and currently_critical:
                status += " | STATUS: EMERGENCY BRAKE"
            elif currently_detected:
                if avoiding_obstacle:
                    status += " | STATUS: AVOIDING OBSTACLE"
                else:
                    status += " | STATUS: STOPPED (obstacle ahead)"
            else:
                status += " | STATUS: Following path"
        fig.suptitle(f"Question 6 Complete Solution (FIXED) - {status}")

        plt.pause(dt)

    print("\nSimulation ended.")
