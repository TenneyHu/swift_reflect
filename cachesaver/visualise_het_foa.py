import re
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import colorsys
from PIL import Image, ImageDraw, ImageFont
import random
import os
from termcolor import colored
import builtins

# these will be given by the user
logs = ''
state_names = {}
states_done_in_puzzle = {}
state_colors = {}

# ---------------------------------------------------------------------------
# IMPORT-ONLY MODE GUARD
# ---------------------------------------------------------------------------
# When this module is imported by other code (e.g. errtst.py) we don't want the
# heavy interactive CLI at the bottom to spin up.  Setting the environment
# variable VISUALISE_HET_FOA_IMPORT_ONLY=1 (done automatically by errtst.py)
# replaces `input` with a stub that immediately raises EOFError, causing the
# CLI loop to exit without side-effects.

if os.environ.get("VISUALISE_HET_FOA_IMPORT_ONLY") == "1":
    builtins.input = lambda *_, **__: (_ for _ in ()).throw(EOFError)

class State(BaseModel):
    name: str
    color: str
    num_thoughts: int
    serial_data: dict
    value: Optional[float] = None
    terminal_data: str = ''


class Timestep(BaseModel):
    timestep: int
    input_states: list[State]
    agent_output_states: list[State]
    state_wins: list[bool]
    state_fails: list[bool]
    replacement_states: list[State]
    values: Optional[list[float]] = None



def generate_distinct_hex_colors(n):
    """
    Generate `n` distinct hex colors that are as different as possible and not close to black.
    
    Returns:
        List of hex color strings (e.g., '#FF5733').
    """
    colors = []
    for i in range(n):
        # Evenly space hues around the color wheel
        hue = i / n
        saturation = 0.65  # Keep saturation high to avoid washed-out colors
        value = 0.8        # Avoid dark (black-ish) colors by setting high brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors


def load_all_logs() -> Dict[str, str]:
    """
    Loads the latest run from both log files for interactive analysis.
    """
    log_contents = {}
    log_files = {
        'with_reflect': 'logs/het_foa_with_reflect.log',
        'no_reflect': 'logs/het_foa.log'
    }
    
    for name, file_path in log_files.items():
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                # Find the last major separator and take everything after it
                last_run = re.split(r'#{50,}', content)[-1].strip()
                if last_run:
                    print(colored(f"Loading {file_path} for interactive analysis.", "cyan"))
                    log_contents[name] = last_run
        except FileNotFoundError:
            # This is not an error, one of the files might not exist yet
            pass
            
    return log_contents


def get_puzzle_idx(log):
    res = re.search(r"het_foa_logs-(\d+)-", log)
    assert res is not None, f'Puzzle index not found in log: {log}'
    return int(res.group(1))


def get_timestep(log):
    res = re.search(r"het_foa_logs-(\d+)-(\d+)", log)
    assert res is not None, f'Timestep not found in log: {log}'
    return int(res.group(2))


def get_py_list(string, type):
    l = eval(string)
    assert isinstance(l, list), f'Expected a list, got {type(l)}: {l}'

    for i, item in enumerate(l):
        l[i] = type(item)

    assert all(isinstance(item, type) for item in l), f'Expected all items to be {type.__name__}, got {l}'
    return l


def get_fleet(log):
    log = log.replace('ValueFunctionWrapped', '').replace('EnvWrapped', '')
    isolated_list = log.split('fleet: ')[-1].strip()
    return get_py_list(isolated_list, str)


def state_name(current_state: str, index):
    if hash(current_state) in state_names:
        return state_names[hash(current_state)]
    
    if index not in states_done_in_puzzle:
        states_done_in_puzzle[index] = 1
    states_done_in_puzzle[index] += 1
    
    idx = states_done_in_puzzle[index]
    idx = len(state_names)
    state_names[hash(current_state)] = f's{idx}'
    return state_names[hash(current_state)]


def get_state_color(state_name: str):
    if state_name in state_colors:
        return state_colors[state_name]
    
    idx = len(state_colors)
    state_colors[state_name] = f'color{idx}'
    return state_colors[state_name]


def get_states_from_log(log):
    index = get_puzzle_idx(log)
    isolated_list = log[log.find('['):]
    states = get_py_list(isolated_list, str)
    
    for i, state in enumerate(states):
        # load python dict from string
        if isinstance(state, str):
            try:
                states[i] = json.loads(state)
            except json.JSONDecodeError:
                raise ValueError(f'Invalid JSON in state: {state}')
    

    for i, state in enumerate(states):
        states[i] = State(
            name=state_name(state['current_state'], index),
            color=get_state_color(state_name(state['current_state'], index)),
            num_thoughts=len(state['reflections']),
            value=state['value'],
            serial_data=state
        )

    return states


def get_timestep_object(logs, timestep=0):
    # assert len(logs) == 6, f'Expected 6 logs for a timestep, got {len(logs)}: {logs}'

    assert re.search(r'het_foa_logs-\d+-\d+-agentinputs', logs[0]), f'First log does not match expected format: {logs[0]}'
    assert re.search(r'het_foa_logs-\d+-\d+-agentouts', logs[1]), f'Second log does not match expected format: {logs[1]}'
    assert re.search(r'het_foa_logs-\d+-\d+-statewins', logs[2]), f'Third log does not match expected format: {logs[2]}'
    if len(logs) > 3: assert re.search(r'het_foa_logs-\d+-\d+-statefails', logs[3]), f'4th log does not match expected format: {logs[3]}'
    if len(logs) > 4: assert re.search(r'het_foa_logs-\d+-\d+-agentreplacements', logs[4]), f'5th log does not match expected format: {logs[4]}'
    if len(logs) > 5: assert re.search(r'het_foa_logs-\d+-\d+-values', logs[5]), f'6th log does not match expected format: {logs[5]}'

    win_list = get_py_list(logs[2].split('statewins: ')[-1].strip(), bool)

    return Timestep(
        timestep=timestep,
        input_states=get_states_from_log(logs[0]),
        agent_output_states=get_states_from_log(logs[1]),
        state_wins=win_list,
        state_fails=get_py_list(logs[3].split('statefails: ')[-1].strip(), bool) if len(logs) > 3 else [False] * len(win_list),
        replacement_states=get_states_from_log(logs[4]) if len(logs) > 4 else [],
        values=get_py_list(logs[5].split('values: ')[-1].strip(), float) if len(logs) > 5 else None
    )


def get_final_timestep(logs, timestep=0):
    assert len(logs) == 5, f'Expected 5 logs for a timestep, got {len(logs)}: {logs}'

    assert re.search(r'het_foa_logs-\d+-\d+-agentinputs', logs[0]), f'First log does not match expected format: {logs[0]}'
    assert re.search(r'het_foa_logs-\d+-\d+-agentouts', logs[1]), f'Second log does not match expected format: {logs[1]}'
    assert re.search(r'het_foa_logs-\d+-\d+-statewins', logs[2]), f'Third log does not match expected format: {logs[2]}'
    assert re.search(r'het_foa_logs-\d+-\d+-statefails', logs[3]), f'4th log does not match expected format: {logs[3]}'
    assert re.search(r'het_foa_logs-\d+-\d+-agentreplacements', logs[4]), f'5th log does not match expected format: {logs[4]}'

    return Timestep(
        timestep=timestep,
        input_states=get_states_from_log(logs[0]),
        agent_output_states=get_states_from_log(logs[1]),
        state_wins=get_py_list(logs[2].split('statewins: ')[-1].strip(), bool),
        state_fails=get_py_list(logs[3].split('statefails: ')[-1].strip(), bool),
        replacement_states=get_states_from_log(logs[4]),
        values=None
    )

def process_log_bundle(logs_str: str):
    global state_names, states_done_in_puzzle, state_colors
    state_names = {}
    states_done_in_puzzle = {}
    state_colors = {}

    logs = logs_str.split('\n')
    het_foa_logs = []
    fleet = []
    for log in logs:
        if 'het_foa_logs' in log:
            if '-fleet:' in log:
                if len(fleet) == 0:
                    fleet = get_fleet(log)
                else:
                    assert fleet == get_fleet(log), f'Fleet mismatch in log: {log} and {fleet=}'
            else:
                het_foa_logs.append(log)

    # Group logs by puzzle and then by timestep
    puzzles_dict = {}
    log_order = ['agentinputs', 'agentouts', 'statewins', 'statefails', 'agentreplacements', 'values']

    def get_log_type(log_line):
        match = re.search(r'-([a-zA-Z]+):', log_line)
        return match.group(1) if match else ""

    for log in het_foa_logs:
        try:
            puzzle_idx = get_puzzle_idx(log)
            timestep_idx = get_timestep(log)

            if puzzle_idx not in puzzles_dict:
                puzzles_dict[puzzle_idx] = {}
            if timestep_idx not in puzzles_dict[puzzle_idx]:
                puzzles_dict[puzzle_idx][timestep_idx] = []
            
            puzzles_dict[puzzle_idx][timestep_idx].append(log)
        except (AssertionError, IndexError):
            # Not a timestep log, ignore for now.
            pass

    graph: Dict[int, List[Timestep]] = {}
    flows = {}
    for puzzle_idx, timesteps_dict in puzzles_dict.items():
        graph[puzzle_idx] = []
        
        # Sort timesteps by index
        sorted_timesteps = sorted(timesteps_dict.items())

        for timestep_idx, logs_for_timestep in sorted_timesteps:
            # Sort logs within a timestep to ensure correct order for processing
            sorted_logs_for_timestep = sorted(logs_for_timestep, key=lambda l: log_order.index(get_log_type(l)) if get_log_type(l) in log_order else -1)
            
            if len(sorted_logs_for_timestep) == 0:
                continue

            # The get_timestep_object is robust enough to handle missing trailing logs
            timestep = get_timestep_object(sorted_logs_for_timestep, timestep_idx)
            graph[puzzle_idx].append(timestep)

        num_colors = len(state_colors)
        colors = generate_distinct_hex_colors(num_colors)
        random.shuffle(colors)

        for k in state_colors:
            state_colors[k] = colors.pop(0)

        # iterate over all States and reset colors
        for timestep in graph[puzzle_idx]:
            for state in timestep.input_states + timestep.agent_output_states + timestep.replacement_states:
                state.color = get_state_color(state.name)

        for timestep in graph[puzzle_idx]:
            for i in range(len(timestep.agent_output_states)):
                if i < len(timestep.state_fails) and timestep.state_fails[i]:
                    timestep.agent_output_states[i].terminal_data = 'Failed'
                elif i < len(timestep.state_wins) and timestep.state_wins[i]:
                    timestep.agent_output_states[i].terminal_data = 'Winning'
                
                if timestep.agent_output_states[i].value is None:
                    if timestep.values and i < len(timestep.values):
                        timestep.agent_output_states[i].value = timestep.values[i]

        if len(fleet) > 0:
            flows[puzzle_idx] = [{
                'agent_name': fleet[i],
                'input_states': [t.input_states[i] for t in graph[puzzle_idx] if len(t.input_states) > i],
                'output_states': [t.agent_output_states[i] for t in graph[puzzle_idx] if len(t.agent_output_states) > i],
            } for i in range(len(fleet))]
            
    return {
        'graph': graph,
        'flows': flows,
        'state_names': state_names,
    }

def get_puzzle_statuses_from_file(log_path):
    try:
        with open(log_path, 'r') as f:
            logs_content = re.split(r'#{50,}', f.read())[-1].strip()
    except FileNotFoundError:
        return {}
    
    _state_names = {}
    _states_done_in_puzzle = {}
    _state_colors = {}

    def _state_name(current_state: str, index):
        if hash(current_state) in _state_names:
            return _state_names[hash(current_state)]
        if index not in _states_done_in_puzzle:
            _states_done_in_puzzle[index] = 1
        _states_done_in_puzzle[index] += 1
        idx = len(_state_names)
        _state_names[hash(current_state)] = f's{idx}'
        return _state_names[hash(current_state)]

    def _get_state_color(state_name_str: str):
        if state_name_str in _state_colors:
            return _state_colors[state_name_str]
        idx = len(_state_colors)
        _state_colors[state_name_str] = f'color{idx}'
        return _state_colors[state_name_str]

    def _get_states_from_log(log):
        index = get_puzzle_idx(log)
        isolated_list = log[log.find('['):]
        states = get_py_list(isolated_list, str)
        
        for i, state in enumerate(states):
            if isinstance(state, str):
                try:
                    states[i] = json.loads(state)
                except json.JSONDecodeError:
                    raise ValueError(f'Invalid JSON in state: {state}')
        
        for i, state in enumerate(states):
            s_name = _state_name(state['current_state'], index)
            states[i] = State(
                name=s_name,
                color=_get_state_color(s_name),
                num_thoughts=len(state['reflections']),
                value=state['value'],
                serial_data=state
            )
        return states
    
    def _get_timestep_object(logs, timestep=0):
        assert re.search(r'het_foa_logs-\d+-\d+-agentinputs', logs[0]), f'First log does not match expected format: {logs[0]}'
        assert re.search(r'het_foa_logs-\d+-\d+-agentouts', logs[1]), f'Second log does not match expected format: {logs[1]}'
        assert re.search(r'het_foa_logs-\d+-\d+-statewins', logs[2]), f'Third log does not match expected format: {logs[2]}'
        if len(logs) > 3: assert re.search(r'het_foa_logs-\d+-\d+-statefails', logs[3]), f'4th log does not match expected format: {logs[3]}'
        if len(logs) > 4: assert re.search(r'het_foa_logs-\d+-\d+-agentreplacements', logs[4]), f'5th log does not match expected format: {logs[4]}'
        if len(logs) > 5: assert re.search(r'het_foa_logs-\d+-\d+-values', logs[5]), f'6th log does not match expected format: {logs[5]}'

        win_list = get_py_list(logs[2].split('statewins: ')[-1].strip(), bool)

        return Timestep(
            timestep=timestep,
            input_states=_get_states_from_log(logs[0]),
            agent_output_states=_get_states_from_log(logs[1]),
            state_wins=win_list,
            state_fails=get_py_list(logs[3].split('statefails: ')[-1].strip(), bool) if len(logs) > 3 else [False] * len(win_list),
            replacement_states=_get_states_from_log(logs[4]) if len(logs) > 4 else [],
            values=get_py_list(logs[5].split('values: ')[-1].strip(), float) if len(logs) > 5 else None
        )

    def _get_final_timestep(logs, timestep=0):
        assert len(logs) == 5, f'Expected 5 logs for a timestep, got {len(logs)}: {logs}'

        assert re.search(r'het_foa_logs-\d+-\d+-agentinputs', logs[0]), f'First log does not match expected format: {logs[0]}'
        assert re.search(r'het_foa_logs-\d+-\d+-agentouts', logs[1]), f'Second log does not match expected format: {logs[1]}'
        assert re.search(r'het_foa_logs-\d+-\d+-statewins', logs[2]), f'Third log does not match expected format: {logs[2]}'
        assert re.search(r'het_foa_logs-\d+-\d+-statefails', logs[3]), f'4th log does not match expected format: {logs[3]}'
        assert re.search(r'het_foa_logs-\d+-\d+-agentreplacements', logs[4]), f'5th log does not match expected format: {logs[4]}'

        return Timestep(
            timestep=timestep,
            input_states=_get_states_from_log(logs[0]),
            agent_output_states=_get_states_from_log(logs[1]),
            state_wins=get_py_list(logs[2].split('statewins: ')[-1].strip(), bool),
            state_fails=get_py_list(logs[3].split('statefails: ')[-1].strip(), bool),
            replacement_states=_get_states_from_log(logs[4]),
            values=None
        )
    
    _logs = logs_content.split('\n')
    _het_foa_logs = []
    _fleet = []
    for log in _logs:
        if 'het_foa_logs' in log:
            if '-fleet:' in log:
                if len(_fleet) == 0:
                    _fleet = get_fleet(log)
                else:
                    assert _fleet == get_fleet(log), f'Fleet mismatch in log: {log} and {_fleet=}'
            else:
                _het_foa_logs.append('het_foa_logs: ' + log.split('het_foa_logs: ')[-1].strip())

    _puzzles = set()
    for log in _het_foa_logs:
        _puzzles.add(get_puzzle_idx(log))

    _puzzles = {
        pid: []
        for pid in list(_puzzles)
    }

    for log in _het_foa_logs:
        puzzle_idx = get_puzzle_idx(log)
        _puzzles[puzzle_idx].append(log)

    _graph: Dict[int, List[Timestep]] = {}
    for puzzle_idx, logs_for_puzzle in _puzzles.items():
        _graph[puzzle_idx] = []
        t = 0
        while len(logs_for_puzzle) > 0:
            if len(logs_for_puzzle) == 5:
                timestep = _get_final_timestep(logs_for_puzzle, t)
                logs_for_puzzle = logs_for_puzzle[5:]
            else:
                timestep = _get_timestep_object(logs_for_puzzle[:6], t)
                logs_for_puzzle = logs_for_puzzle[6:]
            _graph[puzzle_idx].append(timestep)
            t += 1
    
    statuses = {}
    for puzzle_idx, timesteps in _graph.items():
        if timesteps:
                statuses[puzzle_idx] = 'Won' if any(timesteps[-1].state_wins) else 'Failed'
        else:
                statuses[puzzle_idx] = 'Failed'
    
    return statuses

def draw_agent_diagram(agent_name: str, input_states: List[State], output_states: List[State], 
                      x_offset: int = 0, font_size: int = 14) -> tuple[Image.Image, int]:
    """
    Draw a single agent diagram and return the image and the width used.
    """
    # Configuration
    padding = 20
    state_width = 200
    state_padding = 10
    arrow_height = 30
    spacing_between_pairs = 40
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        bold_font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        # Fallback to default font if system fonts aren't available
        font = ImageFont.load_default()
        bold_font = font
    
    # Calculate dimensions
    max_pairs = max(len(input_states), len(output_states))
    
    # Calculate height needed
    agent_name_height = 40
    state_height = 100  # Base height for state rectangles
    total_height = (padding * 2 + 
                   agent_name_height + 
                   max_pairs * (state_height * 2 + arrow_height + spacing_between_pairs))
    
    # Calculate width needed
    diagram_width = state_width + padding * 2
    
    # Create image
    img = Image.new('RGB', (diagram_width, total_height), 'white')
    draw = ImageDraw.Draw(img)
    
    current_y = padding
    
    # Draw agent name in black rectangle
    agent_rect = (x_offset + padding, current_y, 
                  x_offset + padding + state_width, current_y + agent_name_height)
    draw.rectangle(agent_rect, fill='black')
    
    # Center the agent name text
    agent_text_bbox = draw.textbbox((0, 0), agent_name, font=bold_font)
    agent_text_width = agent_text_bbox[2] - agent_text_bbox[0]
    agent_text_height = agent_text_bbox[3] - agent_text_bbox[1]
    agent_text_x = x_offset + padding + (state_width - agent_text_width) // 2
    agent_text_y = current_y + (agent_name_height - agent_text_height) // 2
    draw.text((agent_text_x, agent_text_y), agent_name, fill='white', font=bold_font)
    
    current_y += agent_name_height + padding
    
    # Draw state pairs
    for i in range(max_pairs):
        # Draw input state if exists
        if i < len(input_states):
            current_y = draw_state(draw, input_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        # Draw arrow
        arrow_start_x = x_offset + padding + state_width // 2
        arrow_start_y = current_y + 5
        arrow_end_y = current_y + arrow_height - 5
        
        # Arrow shaft
        draw.line([(arrow_start_x, arrow_start_y), (arrow_start_x, arrow_end_y)], 
                 fill='black', width=2)
        
        # Arrow head
        arrow_head_size = 5
        draw.polygon([(arrow_start_x, arrow_end_y),
                     (arrow_start_x - arrow_head_size, arrow_end_y - arrow_head_size),
                     (arrow_start_x + arrow_head_size, arrow_end_y - arrow_head_size)],
                    fill='black')
        
        current_y += arrow_height
        
        # Draw output state if exists
        if i < len(output_states):
            current_y = draw_state(draw, output_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        current_y += spacing_between_pairs
    
    return img, diagram_width

def draw_state(draw: ImageDraw.Draw, state: State, x: int, y: int, width: int, 
               font: ImageFont.ImageFont, bold_font: ImageFont.ImageFont, padding: int) -> int:
    """
    Draw a single state rectangle and return the y position after drawing.
    """
    # Calculate text lines
    lines = [state.name]  # Bold line
    
    if state.value is not None:
        lines.append(f"Value: {state.value}")
    
    if state.num_thoughts > 0:
        lines.append(f"Thoughts: {state.num_thoughts}")

    if len(state.terminal_data) > 0:
        lines.append(f"{state.terminal_data} State")
    
    # Calculate height needed
    line_height = 20
    text_height = 4 * line_height
    total_height = text_height + padding * 2
    
    # Draw state rectangle
    state_rect = (x, y, x + width, y + total_height)
    draw.rectangle(state_rect, fill=state.color, outline='black', width=1)
    
    # Draw text lines
    text_y = y + padding
    for i, line in enumerate(lines):
        current_font = bold_font if i == 0 else font  # First line (name) is bold
        draw.text((x + padding, text_y), line, fill='black', font=current_font)
        text_y += line_height
    
    return y + total_height

def create_agent_diagrams(diagrams_data: List[dict], spacing: int = 50) -> Image.Image:
    """
    Create multiple agent diagrams in a single image.
    
    diagrams_data: List of dictionaries with keys 'agent_name', 'input_states', 'output_states'
    spacing: Horizontal spacing between diagrams
    """
    if not diagrams_data:
        return Image.new('RGB', (100, 100), 'white')
    
    # First pass: calculate individual diagram dimensions
    diagram_images = []
    diagram_widths = []
    max_height = 0
    
    for data in diagrams_data:
        img, width = draw_agent_diagram(
            data['agent_name'], 
            data['input_states'], 
            data['output_states']
        )
        diagram_images.append(img)
        diagram_widths.append(width)
        max_height = max(max_height, img.height)
    
    # Calculate total width
    total_width = sum(diagram_widths) + spacing * (len(diagrams_data) - 1)
    
    # Create final image
    final_image = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste diagrams
    current_x = 0
    for i, img in enumerate(diagram_images):
        final_image.paste(img, (current_x, 0))
        current_x += diagram_widths[i] + spacing
    
    return final_image

# ------------------------------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------------------------------

# process the logs
log_contents = load_all_logs()
log_data = {}

if 'with_reflect' in log_contents and log_contents['with_reflect']:
    log_data['with_reflect'] = process_log_bundle(log_contents['with_reflect'])
    print(colored("Processed 'with_reflect' logs.", "green"))
if 'no_reflect' in log_contents and log_contents['no_reflect']:
    log_data['no_reflect'] = process_log_bundle(log_contents['no_reflect'])
    print(colored("Processed 'no_reflect' logs.", "green"))

if 'with_reflect' in log_data:
    current_context_name = 'with_reflect'
elif 'no_reflect' in log_data:
    current_context_name = 'no_reflect'
else:
    current_context_name = None

current_puzzle = None
while True:
    prompt_str = '>>> '
    if current_context_name:
        prompt_str = f'({current_context_name}) >>> '
    try:
        cmd = input(prompt_str)
    except EOFError:
        break


    if cmd == 'q':
        break

    if cmd == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        continue

    if cmd == 'switch':
        if current_context_name == 'with_reflect' and 'no_reflect' in log_data:
            current_context_name = 'no_reflect'
            current_puzzle = None
            print(colored("Switched to 'no_reflect' context.", "cyan"))
        elif current_context_name == 'no_reflect' and 'with_reflect' in log_data:
            current_context_name = 'with_reflect'
            current_puzzle = None
            print(colored("Switched to 'with_reflect' context.", "cyan"))
        else:
            print(colored("Cannot switch context. Only one log file loaded.", "yellow"))
        continue

    if not current_context_name:
        print(colored("No logs loaded.", "red"))
        if cmd == 'q':
            break
        continue
    
    graph = log_data[current_context_name]['graph']
    flows = log_data[current_context_name]['flows']
    state_names = log_data[current_context_name]['state_names']


    if cmd == 'compare':
        statuses_no_reflect = get_puzzle_statuses_from_file('logs/het_foa.log')
        statuses_with_reflect = get_puzzle_statuses_from_file('logs/het_foa_with_reflect.log')
        
        all_puzzle_ids = sorted(list(set(statuses_no_reflect.keys()) | set(statuses_with_reflect.keys())))
        
        for puzzle_idx in all_puzzle_ids:
            status_no_reflect = statuses_no_reflect.get(puzzle_idx, 'Not found')
            status_with_reflect = statuses_with_reflect.get(puzzle_idx, 'Not found')
            
            status_no_reflect_colored = colored(status_no_reflect, 'green') if status_no_reflect == 'Won' else colored(status_no_reflect, 'red')
            if status_no_reflect == 'Not found':
                status_no_reflect_colored = colored(status_no_reflect, 'yellow')

            status_with_reflect_colored = colored(status_with_reflect, 'green') if status_with_reflect == 'Won' else colored(status_with_reflect, 'red')
            if status_with_reflect == 'Not found':
                status_with_reflect_colored = colored(status_with_reflect, 'yellow')

            print(f'Puzzle {puzzle_idx}: {status_no_reflect_colored} {colored("->", "yellow")} {status_with_reflect_colored}')
        continue

    if cmd.startswith('open '):
        try:
            puzzle_idx = int(cmd.split(' ')[1])
            if puzzle_idx not in flows:
                print(colored(f'Puzzle {puzzle_idx} not found.', 'red'))
                continue
            
            current_puzzle = puzzle_idx
            print(colored(f'Opened puzzle {puzzle_idx}.', 'green'))
        except (ValueError, IndexError):
            print(colored('Invalid command. Use "open <puzzle_idx>"', 'red'))
        continue

    if cmd.startswith('img'):
        if current_puzzle is None:
            print(colored('No puzzle selected. Use "open <puzzle_idx>" to select a puzzle.', 'red'))
            continue
        
        img = create_agent_diagrams(flows[current_puzzle])
        if current_context_name=="with_reflect":
            os.makedirs('tmp/with_reflect', exist_ok=True)
            img.save(f'tmp/with_reflect/pic_{current_puzzle}.png', format='PNG')  
            print(colored(f'Image saved as tmp/with_reflect/pic_{current_puzzle}.png', 'green'))
        else:
            os.makedirs('tmp/no_reflect', exist_ok=True)
            img.save(f'tmp/no_reflect/pic_{current_puzzle}.png', format='PNG')
            print(colored(f'Image saved as tmp/no_reflect/pic_{current_puzzle}.png', 'green'))
        continue

    if cmd == 'ls':
        for puzzle_idx in flows:
            print(f'Puzzle {puzzle_idx}: ', colored('Won', 'green') if any(graph[puzzle_idx][-1].state_wins) else colored('Failed', 'red'))
        continue

    res = re.search(f'^s(\d+).*$', cmd)
    if res:
        idx = int(res.group(1))
        if current_puzzle is None:
            print(colored('No puzzle selected. Use "open <puzzle_idx>" to select a puzzle.', 'red'))
            continue

        name = f's{idx}'
        if name not in state_names.values():
            print(colored(f'State {name} not found.', 'red'))
            continue

        # Find the state in the current puzzle
        found = False
        state = None
        for timestep in reversed(graph[current_puzzle]):
            for s in timestep.agent_output_states:
                if s.name == name:
                    state = s
                    found = True
                    break

            if found:
                break
        
        
        if not found:
            for s in graph[current_puzzle][0].input_states:
                if s.name == name:
                    state = s
                    found = True
                    break

        if not found:
            print(colored(f'State {name} not found in puzzle {current_puzzle}.', 'red'))
            continue

        attr = cmd.replace(f's{idx}.', '').strip()
        attr = attr.replace('cs', 'current_state') # shorthand
        attr = attr.replace('sd', 'serial_data') # shorthand

        try:
            if any(attr.startswith(field) for field in ['name', 'color', 'num_thoughts', 'value', 'terminal_data', 'serial_data']):
                expr = f'state.{attr}'
            else:
                expr = f'state.serial_data["{attr}"]'
            
            print(eval(expr))
        except:
            print(colored(f'Attribute {attr} not found in state {name}.', 'red'))

        continue


    print(colored('Unknown command.', 'yellow'))