This Python script is an interactive command-line tool for visualizing and inspecting logs from a system named "het_foa". Based on my analysis of the file, here is how you can use it:

### Prerequisites

1.  **Log File**: The script requires a log file located at `logs/het_foa_logs.log`. The script reads from this file to get the data for visualization. You'll need to make sure this file exists and is in the correct format.

2.  **Output Directory**: The script saves generated images to a directory named `tmp`. I'll check if this directory exists, and create it for you if it doesn't.

I will now check for the `tmp` directory.
```bash
ls tmp
```

The `tmp` directory for saving images doesn't exist. I'll create it now.
```bash
mkdir tmp
```


### How to Use the Script

1.  **Run the script** from your terminal:
    ```bash
    python cachesaver/visualise_het_foa.py
    ```

2.  **Use the interactive prompt:**
    Once running, you'll see a `>>>` prompt. Here are the commands you can use:

    *   `ls`: **List available puzzles.** This will show you all the puzzles found in the log file and whether they were won or failed.
        ```
        >>> ls
        Puzzle 0:  Won
        Puzzle 1:  Failed
        ```

    *   `open <puzzle_idx>`: **Select a puzzle.** Use the puzzle number from the `ls` command to open it for inspection.
        ```
        >>> open 0
        ```

    *   `img`: **Generate an image.** After opening a puzzle, this command will create a diagram of the agent and state flows. The image will be saved in the `tmp` directory (e.g., `tmp/pic_0.png`).
        ```
        >>> img
        Image saved as tmp/pic_0.png
        ```

    *   `s<state_id>.<attribute>`: **Inspect a state.** You can look at the details of a specific state (e.g., `s0`, `s1`, which you can find on the generated image).
        *   **Attributes**: `name`, `color`, `num_thoughts`, `value`, `terminal_data`, `serial_data` (or `sd`), `current_state` (or `cs`).
        *   You can also access nested properties inside `serial_data` directly.

        ```
        >>> s1.value
        0.5
        >>> s2.terminal_data
        Winning State
        >>> s3.sd 
        # displays the full serial_data dictionary
        ```

    *   `clear`: Clears your terminal screen.
    *   `q`: Quits the script.

### Example Workflow

1.  Run `python cachesaver/visualise_het_foa.py`.
2.  Type `ls` to see available puzzles.
3.  Type `open 1` (or another puzzle number).
4.  Type `img` to generate the visualization.
5.  Open `tmp/pic_1.png` to view the diagram.
6.  In the diagram, find a state you're interested in, for example `s5`.
7.  Go back to the terminal and type `s5.value` or `s5.serial_data` to get more information about that state.
8.  Type `q` to exit.

Let me know if you have any other questions