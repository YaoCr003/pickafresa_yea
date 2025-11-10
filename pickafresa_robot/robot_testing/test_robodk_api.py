"""
Quick test script to check available RoboDK API methods.
Run this before the main robot_pnp_cli to verify API availability.
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from robolink import Robolink, RUNMODE_SIMULATE
    from robodk import robomath
    print("✓ RoboDK API imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RoboDK API: {e}")
    sys.exit(1)

def test_robodk_api():
    """Test which RoboDK API methods are available."""
    print("\n" + "="*60)
    print("Testing RoboDK API Capabilities")
    print("="*60)
    
    try:
        # Connect to RoboDK
        RDK = Robolink()
        print("✓ Connected to RoboDK")
        
        # Check for test methods
        test_methods = [
            'MoveJ_Test',
            'MoveL_Test',
            'SolveIK_All',
            'setJointConfig',
            'getJointConfig',
            'Collisions',
            'setCollisionActive',
            'getCollisionItems'
        ]
        
        # Get first robot
        robot = RDK.ItemUserPick('Select a robot', filter=2)  # 2 = ITEM_TYPE_ROBOT
        if not robot.Valid():
            print("✗ No robot selected")
            return
        
        print(f"✓ Robot selected: {robot.Name()}")
        
        # Check which methods are available
        print("\nAPI Method Availability:")
        print("-" * 60)
        
        for method in test_methods:
            has_method = hasattr(robot, method) if method not in ['Collisions', 'setCollisionActive', 'getCollisionItems'] else hasattr(RDK, method)
            object_name = "robot" if method not in ['Collisions', 'setCollisionActive', 'getCollisionItems'] else "RDK"
            status = "✓" if has_method else "✗"
            print(f"{status} {object_name}.{method}()")
        
        # Test elbow configuration detection
        print("\nTesting joint configuration detection:")
        print("-" * 60)
        current_joints = robot.Joints()
        print(f"Current joints: {[f'{j:.2f}' for j in current_joints.list()]}")
        
        joint_2 = current_joints.list()[2]
        is_elbow_down = joint_2 < 0
        elbow_status = "elbow-down" if is_elbow_down else "elbow-up"
        print(f"Joint 2 (elbow): {joint_2:.2f} rad -> {elbow_status}")
        
        # Test if SolveIK_All works
        if hasattr(robot, 'SolveIK_All'):
            print("\nTesting SolveIK_All():")
            print("-" * 60)
            current_pose = robot.Pose()
            try:
                all_solutions = robot.SolveIK_All(current_pose)
                if all_solutions:
                    print(f"✓ SolveIK_All() returned {len(all_solutions)} configurations")
                    
                    # Check elbow configurations
                    elbow_up_count = 0
                    elbow_down_count = 0
                    for sol in all_solutions:
                        if sol.list()[2] >= 0:
                            elbow_up_count += 1
                        else:
                            elbow_down_count += 1
                    
                    print(f"  - Elbow-up configurations: {elbow_up_count}")
                    print(f"  - Elbow-down configurations: {elbow_down_count}")
                else:
                    print("✗ SolveIK_All() returned no solutions")
            except Exception as e:
                print(f"✗ SolveIK_All() test failed: {e}")
        
        # Test collision checking
        print("\nTesting collision checking:")
        print("-" * 60)
        try:
            RDK.setCollisionActive(1)
            print("✓ Collision checking enabled")
            
            collision_count = RDK.Collisions()
            print(f"✓ Current collision count: {collision_count}")
            
            if hasattr(RDK, 'getCollisionItems'):
                try:
                    items = RDK.getCollisionItems()
                    print(f"✓ getCollisionItems() available (returned {len(items) if items else 0} items)")
                except:
                    print("✗ getCollisionItems() available but returned error")
            else:
                print("✗ getCollisionItems() not available (will use fallback)")
        except Exception as e:
            print(f"✗ Collision checking test failed: {e}")
        
        # Test movement test methods
        if hasattr(robot, 'MoveJ_Test'):
            print("\nTesting MoveJ_Test():")
            print("-" * 60)
            try:
                # Test with current position (should be safe)
                temp_target = RDK.AddTarget("_test_target")
                temp_target.setPose(robot.Pose())
                result = robot.MoveJ_Test(temp_target)
                temp_target.Delete()
                print(f"✓ MoveJ_Test() available (result: {result})")
                print(f"  (0 = OK, negative = collision/error)")
            except Exception as e:
                print(f"✗ MoveJ_Test() test failed: {e}")
        
        print("\n" + "="*60)
        print("API Test Complete")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robodk_api()
