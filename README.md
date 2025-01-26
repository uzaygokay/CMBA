# CMBA

Cognitive Modelling of Biological Agents

## Generating Mazes

1. Run the file `maze_generator.py`
2. Set the desired maze size in rows and columns
3. Set the agent's starting location. Re-clicking the agent will change the direction that the agent is facing in.
4. Set the target
5. Add more walls if necessary.
6. (Advanced) Add rewards and dangers.
7. Export the maze

Exported maze legend:

```
    W = Wall
    S = Start
    T = Target
    D = Danger
    R = Reward
    0 = Open space
```

## Running the environment

1. In the main function,
   1. Modify `maze_file = "demo_maze.txt"` to read your own maze.
   2. Use cognitive modelling techniques to come up with an action sequence of your own and modify this line of code: `agent_action_sequence = demo_actions_set_1`
2. Run `maze_env.py`
