"""
Tests for the Robot Simulation Module.
"""

import pytest
from src.simulation.robot_sim import (
    RobotSimulator,
    RobotState,
    SimulationResult,
    GridWorld,
    Direction
)


class TestRobotState:
    """Tests for RobotState."""
    
    def test_initial_state(self):
        state = RobotState()
        assert state.x == 0
        assert state.y == 0
        assert state.direction == Direction.NORTH
        assert state.holding is None
    
    def test_position_property(self):
        state = RobotState(x=3, y=5)
        assert state.position == (3, 5)
    
    def test_move_forward_north(self):
        state = RobotState(x=0, y=0, direction=Direction.NORTH)
        state.move_forward()
        assert state.position == (0, 1)
    
    def test_move_forward_east(self):
        state = RobotState(x=0, y=0, direction=Direction.EAST)
        state.move_forward()
        assert state.position == (1, 0)
    
    def test_move_forward_south(self):
        state = RobotState(x=1, y=1, direction=Direction.SOUTH)
        state.move_forward()
        assert state.position == (1, 0)
    
    def test_move_forward_west(self):
        state = RobotState(x=1, y=1, direction=Direction.WEST)
        state.move_forward()
        assert state.position == (0, 1)
    
    def test_turn_left(self):
        state = RobotState(direction=Direction.NORTH)
        state.turn_left()
        assert state.direction == Direction.WEST
        state.turn_left()
        assert state.direction == Direction.SOUTH
        state.turn_left()
        assert state.direction == Direction.EAST
        state.turn_left()
        assert state.direction == Direction.NORTH
    
    def test_turn_right(self):
        state = RobotState(direction=Direction.NORTH)
        state.turn_right()
        assert state.direction == Direction.EAST
        state.turn_right()
        assert state.direction == Direction.SOUTH
        state.turn_right()
        assert state.direction == Direction.WEST
        state.turn_right()
        assert state.direction == Direction.NORTH
    
    def test_history_tracking(self):
        state = RobotState(x=0, y=0)
        state.move_forward()
        state.move_forward()
        assert len(state.history) == 3  # Initial + 2 moves
        assert (0, 0) in state.history
        assert (0, 1) in state.history
        assert (0, 2) in state.history
    
    def test_propositions(self):
        state = RobotState()
        state.set_proposition("at_goal", True)
        assert state.has_proposition("at_goal")
        
        state.set_proposition("at_goal", False)
        assert not state.has_proposition("at_goal")


class TestGridWorld:
    """Tests for GridWorld."""
    
    def test_default_creation(self):
        world = GridWorld()
        assert world.width == 10
        assert world.height == 10
    
    def test_custom_dimensions(self):
        world = GridWorld(width=5, height=8)
        assert world.width == 5
        assert world.height == 8
    
    def test_valid_position(self):
        world = GridWorld(width=5, height=5)
        assert world.is_valid_position(0, 0)
        assert world.is_valid_position(4, 4)
        assert not world.is_valid_position(-1, 0)
        assert not world.is_valid_position(0, -1)
        assert not world.is_valid_position(5, 0)
        assert not world.is_valid_position(0, 5)
    
    def test_obstacles(self):
        world = GridWorld(width=5, height=5, obstacles={(2, 2), (3, 3)})
        assert not world.is_valid_position(2, 2)
        assert not world.is_valid_position(3, 3)
        assert world.is_valid_position(1, 1)
    
    def test_add_obstacle(self):
        world = GridWorld()
        world.add_obstacle(3, 3)
        assert not world.is_valid_position(3, 3)
    
    def test_goals(self):
        world = GridWorld(goals={"goal": (5, 5)})
        assert world.goals["goal"] == (5, 5)
        assert world.get_cell_type(5, 5) == "goal:goal"
    
    def test_add_goal(self):
        world = GridWorld()
        world.add_goal("target", 7, 7)
        assert world.goals["target"] == (7, 7)
    
    def test_get_neighbors(self):
        world = GridWorld(width=3, height=3)
        # Center has 4 neighbors
        neighbors = world.get_neighbors(1, 1)
        assert len(neighbors) == 4
        
        # Corner has 2 neighbors
        neighbors = world.get_neighbors(0, 0)
        assert len(neighbors) == 2
    
    def test_from_ascii(self):
        ascii_map = """
....G
.....
..#..
.....
R....
        """
        world = GridWorld.from_ascii(ascii_map)
        # Obstacle at (2, 2) after flipping Y
        assert len(world.obstacles) >= 1
        assert "goal" in world.goals or len(world.goals) >= 1


class TestRobotSimulator:
    """Tests for RobotSimulator."""
    
    @pytest.fixture
    def simple_world(self):
        world = GridWorld(width=5, height=5)
        world.add_goal("goal", 4, 4)
        return world
    
    @pytest.fixture
    def simulator(self, simple_world):
        return RobotSimulator(simple_world)
    
    def test_reset(self, simulator):
        simulator.reset(x=2, y=2, direction=Direction.EAST)
        assert simulator.state.position == (2, 2)
        assert simulator.state.direction == Direction.EAST
        assert simulator.step_count == 0
    
    def test_execute_move_forward(self, simulator):
        simulator.reset()
        success = simulator.execute_action("move_forward")
        assert success
        assert simulator.state.position == (0, 1)
    
    def test_execute_turn_left(self, simulator):
        simulator.reset()
        success = simulator.execute_action("turn_left")
        assert success
        assert simulator.state.direction == Direction.WEST
    
    def test_execute_turn_right(self, simulator):
        simulator.reset()
        success = simulator.execute_action("turn_right")
        assert success
        assert simulator.state.direction == Direction.EAST
    
    def test_execute_stop(self, simulator):
        simulator.reset()
        success = simulator.execute_action("stop")
        assert success
    
    def test_blocked_movement(self):
        world = GridWorld(width=3, height=3, obstacles={(1, 0)})
        sim = RobotSimulator(world)
        sim.reset(direction=Direction.EAST)
        
        success = sim.execute_action("move_forward")
        assert not success  # Blocked by obstacle
        assert sim.state.position == (0, 0)  # Didn't move
    
    def test_run_simple_commands(self, simulator):
        commands = ["move_forward", "move_forward", "turn_right", "move_forward"]
        result = simulator.run_simple_commands(commands)
        
        assert isinstance(result, SimulationResult)
        assert result.steps == 4
        assert result.final_state.position == (1, 2)
    
    def test_trace_recording(self, simulator):
        simulator.reset()
        simulator.execute_action("move_forward")
        simulator.execute_action("turn_right")
        
        assert len(simulator.trace) == 3  # init + 2 actions
        assert simulator.trace[0]["action"] == "init"
        assert simulator.trace[1]["action"] == "move_forward"
        assert simulator.trace[2]["action"] == "turn_right"
    
    def test_ascii_visualization(self, simulator):
        simulator.reset()
        simulator.execute_action("move_forward")
        
        viz = simulator.get_ascii_visualization()
        assert isinstance(viz, str)
        assert "^" in viz  # Robot facing north


class TestSimulationResult:
    """Tests for SimulationResult."""
    
    def test_success_result(self):
        state = RobotState(x=4, y=4)
        result = SimulationResult(
            success=True,
            steps=10,
            final_state=state,
            trace=[],
            goals_reached=["goal"]
        )
        assert result.success
        assert "SUCCESS" in str(result)
    
    def test_failed_result(self):
        state = RobotState()
        result = SimulationResult(
            success=False,
            steps=5,
            final_state=state,
            trace=[],
            violations=["G(!obstacle) violated"]
        )
        assert not result.success
        assert "FAILED" in str(result)


class TestRunController:
    """Tests for running synthesized controllers."""
    
    @pytest.fixture
    def simulator(self):
        world = GridWorld(width=5, height=5)
        world.add_goal("goal", 4, 0)
        return RobotSimulator(world)
    
    def test_simple_controller(self, simulator):
        controller = {
            "states": ["s0", "s1", "s2"],
            "initial_state": "s0",
            "accepting_states": ["s2"],
            "transitions": [
                {"from": "s0", "to": "s1", "action": "turn_right"},
                {"from": "s1", "to": "s1", "action": "move_forward"},
                {"from": "s1", "to": "s2", "action": "stop"},
            ],
            "guarantees": []
        }
        
        result = simulator.run_controller(controller, max_steps=10)
        assert isinstance(result, SimulationResult)
        assert result.steps <= 10
    
    def test_controller_max_steps(self, simulator):
        # Controller that loops forever
        controller = {
            "states": ["s0"],
            "initial_state": "s0",
            "accepting_states": [],
            "transitions": [
                {"from": "s0", "to": "s0", "action": "turn_right"},
            ],
            "guarantees": []
        }
        
        result = simulator.run_controller(controller, max_steps=5)
        assert result.steps == 5
