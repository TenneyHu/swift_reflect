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

logs = ''
state_names = {}
states_done_in_puzzle = {}
state_colors = {}

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
        'reflect_summary': 'logs/reflect_summary.log',
        'reflect_prevk': 'logs/reflect_prevk.log'
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

def get_task_and_puzzle_idx(log):
    res = re.search(r"reflect_(?:summary|prevk)_logs-([^-]+)-(\d+)-", log)
    assert res is not None, f'Task and puzzle index not found in log: {log}'
    return res.group(1), int(res.group(2))

def get_task_puzzle_timestep(log):
    res = re.search(r"reflect_(?:summary|prevk)_logs-([^-]+)-(\d+)-(\d+)", log)
    assert res is not None, f'Task, puzzle index, or timestep not found in log: {log}'
    return res.group(1), int(res.group(2)), int(res.group(3))

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
        states_done_in_puzzle[index] = 0
    states_done_in_puzzle[index] += 1
    
    idx = states_done_in_puzzle[index]
    state_names[hash(current_state)] = f's{idx}'
    return state_names[hash(current_state)]

def get_state_color(state_name: str):
    if state_name in state_colors:
        return state_colors[state_name]
    
    idx = len(state_colors)
    state_colors[state_name] = f'color{idx}'
    return state_colors[state_name]

def get_states_from_log(log):
    _, puzzle_idx = get_task_and_puzzle_idx(log)
    isolated_list = log[log.find('['):]
    states_str = get_py_list(isolated_list, str)
    
    parsed_states = []
    for s_str in states_str:
        try:
            parsed_states.append(json.loads(s_str))
        except json.JSONDecodeError:
            raise ValueError(f'Invalid JSON in state: {s_str}')

    states_out = []
    for state_data in parsed_states:
        s_name = state_name(state_data['current_state'], puzzle_idx)
        states_out.append(State(
            name=s_name,
            color=get_state_color(s_name),
            num_thoughts=len(state_data['reflections']),
            value=state_data.get('value'),
            serial_data=state_data
        ))

    return states_out

def get_timestep_object(logs, timestep=0):
    log_map = {}
    for log in logs:
        match = re.search(r'-(agentinputs|agentouts|statewins|statefails|reflections|summaries):', log)
        if match:
            log_map[match.group(1)] = log

    input_states = get_states_from_log(log_map['agentinputs'])
    output_states = get_states_from_log(log_map['agentouts'])
    state_wins = get_py_list(log_map['statewins'].split('statewins: ')[-1].strip(), bool)
    state_fails = get_py_list(log_map['statefails'].split('statefails: ')[-1].strip(), bool)

    if 'reflections' in log_map:
        reflections_list = get_py_list(log_map['reflections'].split('reflections: ')[-1].strip(), int)
        for i, num_reflections in enumerate(reflections_list):
            if i < len(output_states):
                output_states[i].num_thoughts = num_reflections
    
    if 'summaries' in log_map:
        summaries_list = get_py_list(log_map['summaries'].split('summaries: ')[-1].strip(), list)
        for i, summary in enumerate(summaries_list):
            if i < len(output_states):
                output_states[i].num_thoughts = len(summary)

    return Timestep(
        timestep=timestep,
        input_states=input_states,
        agent_output_states=output_states,
        state_wins=state_wins,
        state_fails=state_fails,
        replacement_states=[],
        values=None,
    )

def process_log_bundle(logs_str: str):
    global state_names, states_done_in_puzzle, state_colors
    state_names = {}
    states_done_in_puzzle = {}
    state_colors = {}

    logs = logs_str.split('\n')
    
    fleet_dict = {}
    puzzle_logs = []
    log_prefix = ''
    if 'reflect_summary_logs' in logs_str:
        log_prefix = 'reflect_summary_logs'
    elif 'reflect_prevk_logs' in logs_str:
        log_prefix = 'reflect_prevk_logs'

    for log in logs:
        if log_prefix in log:
            if '-fleet:' in log:
                task_name, _ = get_task_and_puzzle_idx(log)
                if task_name not in fleet_dict:
                    fleet_dict[task_name] = get_fleet(log)
            else:
                puzzle_logs.append(log)

    tasks_dict = {}
    log_order = ['agentinputs', 'agentouts', 'statewins', 'statefails', 'reflections', 'summaries']

    def get_log_type(log_line):
        match = re.search(r'-([a-zA-Z]+):', log_line)
        return match.group(1) if match else ""

    for log in puzzle_logs:
        try:
            task_name, puzzle_idx, timestep_idx = get_task_puzzle_timestep(log)

            if task_name not in tasks_dict:
                tasks_dict[task_name] = {}
            if puzzle_idx not in tasks_dict[task_name]:
                tasks_dict[task_name][puzzle_idx] = {}
            if timestep_idx not in tasks_dict[task_name][puzzle_idx]:
                tasks_dict[task_name][puzzle_idx][timestep_idx] = []
            
            tasks_dict[task_name][puzzle_idx][timestep_idx].append(log)
        except (AssertionError, IndexError):
            pass

    tasks_data = {}
    for task_name, puzzles_dict in tasks_dict.items():
        graph: Dict[int, List[Timestep]] = {}
        flows = {}
        for puzzle_idx, timesteps_dict in puzzles_dict.items():
            graph[puzzle_idx] = []
            
            sorted_timesteps = sorted(timesteps_dict.items())

            for timestep_idx, logs_for_timestep in sorted_timesteps:
                sorted_logs_for_timestep = sorted(logs_for_timestep, key=lambda l: log_order.index(get_log_type(l)) if get_log_type(l) in log_order else -1)
                
                if len(sorted_logs_for_timestep) < 4:
                    continue

                timestep = get_timestep_object(sorted_logs_for_timestep, timestep_idx)
                graph[puzzle_idx].append(timestep)

        num_colors = len(state_colors)
        colors = generate_distinct_hex_colors(num_colors)
        random.shuffle(colors)

        for k in state_colors:
            state_colors[k] = colors.pop(0)

        for puzzle_idx in graph:
            for timestep in graph[puzzle_idx]:
                for state in timestep.input_states + timestep.agent_output_states:
                    state.color = get_state_color(state.name)

        for puzzle_idx in graph:
            for timestep in graph[puzzle_idx]:
                for i in range(len(timestep.agent_output_states)):
                    if i < len(timestep.state_wins) and timestep.state_wins[i]:
                        timestep.agent_output_states[i].terminal_data = 'Winning'
                    elif i < len(timestep.state_fails) and timestep.state_fails[i]:
                        timestep.agent_output_states[i].terminal_data = 'Failed'

        if task_name in fleet_dict and len(fleet_dict[task_name]) > 0:
            fleet = fleet_dict[task_name]
            for puzzle_idx in graph:
                flows[puzzle_idx] = [{
                    'agent_name': fleet[0],
                    'input_states': [t.input_states[i] for t in graph[puzzle_idx] if len(t.input_states) > i],
                    'output_states': [t.agent_output_states[i] for t in graph[puzzle_idx] if len(t.agent_output_states) > i],
                } for i in range(1)]
        
        tasks_data[task_name] = {
            'graph': graph,
            'flows': flows,
            'state_names': state_names,
        }
            
    return tasks_data

def get_puzzle_statuses_from_file(log_path):
    try:
        with open(log_path, 'r') as f:
            logs_content = re.split(r'#{50,}', f.read())[-1].strip()
        if not logs_content:
            return {}
    except FileNotFoundError:
        return {}
    
    tasks_data = process_log_bundle(logs_content)
    task_statuses = {}

    for task_name, data in tasks_data.items():
        graph = data['graph']
        statuses = {}
        for puzzle_idx, timesteps in graph.items():
            if timesteps:
                statuses[puzzle_idx] = 'Won' if any(timesteps[-1].state_wins) else 'Failed'
            else:
                statuses[puzzle_idx] = 'Failed'
        task_statuses[task_name] = statuses
    
    return task_statuses

def draw_agent_diagram(agent_name: str, input_states: List[State], output_states: List[State], 
                      x_offset: int = 0, font_size: int = 14) -> tuple[Image.Image, int]:
    padding = 20
    state_width = 200
    state_padding = 10
    arrow_height = 30
    spacing_between_pairs = 40
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        bold_font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        font = ImageFont.load_default()
        bold_font = font
    
    max_pairs = max(len(input_states), len(output_states))
    
    agent_name_height = 40
    state_height = 100
    total_height = (padding * 2 + 
                   agent_name_height + 
                   max_pairs * (state_height * 2 + arrow_height + spacing_between_pairs))
    
    diagram_width = state_width + padding * 2
    
    img = Image.new('RGB', (diagram_width, total_height), 'white')
    draw = ImageDraw.Draw(img)
    
    current_y = padding
    
    agent_rect = (x_offset + padding, current_y, 
                  x_offset + padding + state_width, current_y + agent_name_height)
    draw.rectangle(agent_rect, fill='black')
    
    agent_text_bbox = draw.textbbox((0, 0), agent_name, font=bold_font)
    agent_text_width = agent_text_bbox[2] - agent_text_bbox[0]
    agent_text_height = agent_text_bbox[3] - agent_text_bbox[1]
    agent_text_x = x_offset + padding + (state_width - agent_text_width) // 2
    agent_text_y = current_y + (agent_name_height - agent_text_height) // 2
    draw.text((agent_text_x, agent_text_y), agent_name, fill='white', font=bold_font)
    
    current_y += agent_name_height + padding
    
    for i in range(max_pairs):
        if i < len(input_states):
            current_y = draw_state(draw, input_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        arrow_start_x = x_offset + padding + state_width // 2
        arrow_start_y = current_y + 5
        arrow_end_y = current_y + arrow_height - 5
        
        draw.line([(arrow_start_x, arrow_start_y), (arrow_start_x, arrow_end_y)], 
                 fill='black', width=2)
        
        arrow_head_size = 5
        draw.polygon([(arrow_start_x, arrow_end_y),
                     (arrow_start_x - arrow_head_size, arrow_end_y - arrow_head_size),
                     (arrow_start_x + arrow_head_size, arrow_end_y - arrow_head_size)],
                    fill='black')
        
        current_y += arrow_height
        
        if i < len(output_states):
            current_y = draw_state(draw, output_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        current_y += spacing_between_pairs
    
    return img.crop((0,0,diagram_width, current_y)), diagram_width

def draw_state(draw: ImageDraw.Draw, state: State, x: int, y: int, width: int, 
               font: ImageFont.ImageFont, bold_font: ImageFont.ImageFont, padding: int) -> int:
    lines = [state.name]
    
    if state.value is not None:
        lines.append(f"Value: {state.value}")
    
    if state.num_thoughts > 0:
        lines.append(f"Thoughts: {state.num_thoughts}")

    if len(state.terminal_data) > 0:
        lines.append(f"{state.terminal_data} State")
    
    line_height = 20
    text_height = 4 * line_height
    total_height = text_height + padding * 2
    
    state_rect = (x, y, x + width, y + total_height)
    draw.rectangle(state_rect, fill=state.color, outline='black', width=1)
    
    text_y = y + padding
    for i, line in enumerate(lines):
        current_font = bold_font if i == 0 else font
        draw.text((x + padding, text_y), line, fill='black', font=current_font)
        text_y += line_height
    
    return y + total_height

def create_agent_diagrams(diagrams_data: List[dict], spacing: int = 50) -> Image.Image:
    if not diagrams_data:
        return Image.new('RGB', (100, 100), 'white')
    
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
    
    total_width = sum(diagram_widths) + spacing * (len(diagrams_data) - 1)
    
    final_image = Image.new('RGB', (total_width, max_height), 'white')
    
    current_x = 0
    for i, img in enumerate(diagram_images):
        final_image.paste(img, (current_x, 0))
        current_x += diagram_widths[i] + spacing
    
    return final_image

# ------------------------------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------------------------------

log_contents = load_all_logs()
log_data = {}

if 'reflect_summary' in log_contents and log_contents['reflect_summary']:
    log_data['reflect_summary'] = process_log_bundle(log_contents['reflect_summary'])
    print(colored("Processed 'reflect_summary' logs.", "green"))
if 'reflect_prevk' in log_contents and log_contents['reflect_prevk']:
    log_data['reflect_prevk'] = process_log_bundle(log_contents['reflect_prevk'])
    print(colored("Processed 'reflect_prevk' logs.", "green"))

if 'reflect_summary' in log_data:
    current_context_name = 'reflect_summary'
elif 'reflect_prevk' in log_data:
    current_context_name = 'reflect_prevk'
else:
    current_context_name = None

current_task = None
current_puzzle = None
while True:
    prompt_str = '>>> '
    if current_context_name:
        prompt_str = f'({current_context_name})'
        if current_task:
            prompt_str += f'/{current_task}'
    prompt_str += ' >>> '

    try:
        cmd = input(prompt_str)
    except EOFError:
        break

    if cmd == 'q':
        break
    
    if cmd == 'back':
        if current_puzzle is not None:
            current_puzzle = None
        elif current_task is not None:
            current_task = None
        continue

    if cmd == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        continue

    if cmd == 'switch':
        if current_context_name == 'reflect_summary' and 'reflect_prevk' in log_data:
            current_context_name = 'reflect_prevk'
            current_puzzle = None
            current_task = None
            print(colored("Switched to 'reflect_prevk' context.", "cyan"))
        elif current_context_name == 'reflect_prevk' and 'reflect_summary' in log_data:
            current_context_name = 'reflect_summary'
            current_puzzle = None
            current_task = None
            print(colored("Switched to 'reflect_summary' context.", "cyan"))
        else:
            print(colored("Cannot switch context. Only one log file loaded.", "yellow"))
        continue

    if not current_context_name:
        print(colored("No logs loaded.", "red"))
        if cmd == 'q':
            break
        continue
    
    tasks_data = log_data[current_context_name]

    if cmd == 'compare':
        statuses_summary = get_puzzle_statuses_from_file('logs/reflect_summary.log')
        statuses_prevk = get_puzzle_statuses_from_file('logs/reflect_prevk.log')
        
        all_task_names = sorted(list(set(statuses_summary.keys()) | set(statuses_prevk.keys())))

        for task_name in all_task_names:
            print(f"\n--- Task: {task_name} ---")
            task_summary = statuses_summary.get(task_name, {})
            task_prevk = statuses_prevk.get(task_name, {})
            
            all_puzzle_ids = sorted(list(set(task_summary.keys()) | set(task_prevk.keys())))
            
            print(f"{'Puzzle':<10}{'reflect_summary':<20}{'reflect_prevk':<20}")
            print(f"{'-'*8:<10}{'-'*15:<20}{'-'*15:<20}")

            for puzzle_idx in all_puzzle_ids:
                status_summary = task_summary.get(puzzle_idx, 'Not found')
                status_prevk = task_prevk.get(puzzle_idx, 'Not found')
                
                status_summary_colored = colored(status_summary, 'green') if status_summary == 'Won' else colored(status_summary, 'red')
                if status_summary == 'Not found':
                    status_summary_colored = colored(status_summary, 'yellow')

                status_prevk_colored = colored(status_prevk, 'green') if status_prevk == 'Won' else colored(status_prevk, 'red')
                if status_prevk == 'Not found':
                    status_prevk_colored = colored(status_prevk, 'yellow')

                print(f"{puzzle_idx:<10}{status_summary_colored:<28}{status_prevk_colored:<28}")
        continue

    if cmd.startswith('open '):
        try:
            if current_task is None:
                task_name = cmd.split(' ')[1]
                if task_name not in tasks_data:
                    print(colored(f'Task {task_name} not found.', 'red'))
                    continue
                current_task = task_name
                print(colored(f'Opened task {task_name}.', 'green'))
            else:
                puzzle_idx = int(cmd.split(' ')[1])
                if puzzle_idx not in tasks_data[current_task]['flows']:
                    print(colored(f'Puzzle {puzzle_idx} not found in task {current_task}.', 'red'))
                    continue
                current_puzzle = puzzle_idx
                print(colored(f'Opened puzzle {puzzle_idx}.', 'green'))
        except (ValueError, IndexError):
            print(colored('Invalid command. Use "open <task_name>" or "open <puzzle_idx>"', 'red'))
        continue

    if cmd.startswith('img'):
        if current_task is None:
            print(colored('No task selected. Use "open <task_name>" to select a task.', 'red'))
            continue
        if current_puzzle is None:
            print(colored('No puzzle selected. Use "open <puzzle_idx>" to select a puzzle.', 'red'))
            continue
        
        flows = tasks_data[current_task]['flows']
        img = create_agent_diagrams(flows[current_puzzle])
        
        os.makedirs(f'tmp/{current_context_name}/{current_task}', exist_ok=True)
        img_path = f'tmp/{current_context_name}/{current_task}/pic_{current_puzzle}.png'
        img.save(img_path, format='PNG')  
        print(colored(f'Image saved as {img_path}', 'green'))
        continue

    if cmd == 'ls':
        if current_task is None:
            print("Available tasks:")
            for task_name in tasks_data:
                print(f"- {task_name}")
        else:
            graph = tasks_data[current_task]['graph']
            flows = tasks_data[current_task]['flows']
            for puzzle_idx in flows:
                print(f'Puzzle {puzzle_idx}: ', colored('Won', 'green') if any(graph[puzzle_idx][-1].state_wins) else colored('Failed', 'red'))
        continue

    res = re.search(f'^s(\d+.*$)', cmd)
    if res:
        if current_task is None:
            print(colored('No task selected. Use "open <task_name>" to select a task.', 'red'))
            continue
        if current_puzzle is None:
            print(colored('No puzzle selected. Use "open <puzzle_idx>" to select a puzzle.', 'red'))
            continue
        
        cmd_parts = res.group(1).split('.')
        state_id = "s" + cmd_parts[0]

        state_names_map = tasks_data[current_task]['state_names']
        if state_id not in state_names_map.values():
            print(colored(f'State {state_id} not found.', 'red'))
            continue

        graph = tasks_data[current_task]['graph']
        found_state = None
        for timestep in reversed(graph[current_puzzle]):
            for s in timestep.agent_output_states + timestep.input_states:
                if s.name == state_id:
                    found_state = s
                    break
            if found_state:
                break
        
        if not found_state:
             print(colored(f'State {state_id} not found in puzzle {current_puzzle}.', 'red'))
             continue

        if len(cmd_parts) > 1:
            attr = cmd_parts[1].strip()
            attr = attr.replace('cs', 'current_state') 
            attr = attr.replace('sd', 'serial_data')

            try:
                if hasattr(found_state, attr):
                    print(getattr(found_state, attr))
                elif attr in found_state.serial_data:
                    print(found_state.serial_data[attr])
                else:
                    print(colored(f'Attribute {attr} not found in state {state_id}.', 'red'))
            except Exception as e:
                print(colored(f'Error accessing attribute {attr}: {e}', 'red'))
        else:
            print(found_state)

        continue

    print(colored('Unknown command. Type "help" for a list of commands.', 'yellow'))


