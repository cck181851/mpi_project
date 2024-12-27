from mpi4py import MPI
import math
import sys
import threading
import time

# UNIT class Definition
class UNIT:
    def __init__(self, name, hp, attack_power):
        self.name = name
        self.hp = hp
        self.attack_power = attack_power
    def __repr__(self):
        return self.name
    
# SharedData class for RPC calls
class SharedData:
    def __init__(self, sub_block, received_boundaries):
        self.sub_block = sub_block  
        self.received_boundaries = received_boundaries  
        self.lock = threading.Lock()  

# Faction statistics
faction_stats = {
    'E': {
        'heal_rate': 3,
        'attack_power': 2,
        'max_health': 18,
        'attack_pattern': [(-1, 0), (1, 0), (0, -1), (0, 1)]
    },
    'F': {
        'heal_rate': 1,
        'attack_power': 4,
        'max_attack_power': 6,
        'max_health': 12,
        'attack_pattern': [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1), (1, 0), (1, 1)]
    },
    'W': {
        'heal_rate': 2,
        'attack_power': 3,
        'max_health': 14,
        'attack_pattern': [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    },
    'A': {
        'heal_rate': 2,
        'attack_power': 2,
        'max_health': 10,
        'attack_pattern': [(-2, -2),         (-2, 0)   ,(-2, 2),
                                     (-1, -1), (-1, 0), (-1, 1),
                           (0, -2),  (0, -1),           (0, 1),  (0, 2),
                                     (1, -1),  (1, 0),  (1, 1), 
                           (2, -2),          (2, 0),          (2, 2)]
    }
}
NUM_LAYERS = 3          # Number of layers to exchange
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processors = comm.Get_size()

# communication tags
TAG_INIT = 0
TAG_MOVEMENT_PHASE = 1
TAG_ATTACK_PHASE = 2
TAG_RPC_REQUEST = 10
TAG_RPC_RESPONSE = 11
TAG_TERMINATE = 99

def parse_input(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    N, W, T, R = map(int, lines[0].strip().split())
    waves = {}
    wave_num = 0
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("Wave"):
            wave_num += 1
            waves[wave_num] = {'E': [], 'F': [], 'W': [], 'A': []}
        elif line.startswith("E:"):
            waves[wave_num]['E'] = parse_coordinates(line[2:])
        elif line.startswith("F:"):
            waves[wave_num]['F'] = parse_coordinates(line[2:])
        elif line.startswith("W:"):
            waves[wave_num]['W'] = parse_coordinates(line[2:])
        elif line.startswith("A:"):
            waves[wave_num]['A'] = parse_coordinates(line[2:])
    return N, W, T, R, waves

def parse_coordinates(coord_line):
    if coord_line.strip() == '':
        return []
    coords = coord_line.split(',')
    return [tuple(map(int, coord.strip().split())) for coord in coords]

def populate_grid(grid, N, wave_units):
    for faction, positions in wave_units.items():
        for x, y in positions:
            unit = UNIT(faction, faction_stats[faction]['max_health'], faction_stats[faction]['attack_power'])
            if grid[x][y].name == '.':
                grid[x][y] = unit

def split_grid(grid, P):
    sqrt_P = int(math.sqrt(P - 1))
    if sqrt_P ** 2 != (P - 1):
        if rank == 0:
            print("Number of worker processors must be a perfect square (P-1).")
        sys.exit()
    block_size = len(grid) // sqrt_P
    blocks = []
    for i in range(sqrt_P):
        for j in range(sqrt_P):
            block = [row[j*block_size:(j+1)*block_size] for row in grid[i*block_size:(i+1)*block_size]]
            blocks.append((block, (i, j)))
    return blocks

def get_global_idx(x, y, rank, sub_block_size, sqrt_P, N):
    sub_block_row = (rank - 1) // sqrt_P
    sub_block_col = (rank - 1) % sqrt_P
    global_origin_x = sub_block_row * sub_block_size
    global_origin_y = sub_block_col * sub_block_size
    global_x = global_origin_x + x
    global_y = global_origin_y + y
    return global_x, global_y

#  visualize the grid
def visualise(grid):
    for row in grid:
        cur = [col.name for col in row]
        print(" ".join(cur))
    print()

def get_neighbors(rank, sqrt_P):
    i = (rank - 1) // sqrt_P
    j = (rank - 1) % sqrt_P
    neighbors = {}
    directions = {
        'N': (-1, 0),
        'NE': (-1, 1),
        'E': (0, 1),
        'SE': (1, 1),
        'S': (1, 0),
        'SW': (1, -1),
        'W': (0, -1),
        'NW': (-1, -1)
    }
    for direction, (di, dj) in directions.items():
        ni, nj = i + di, j + dj
        if 0 <= ni < sqrt_P and 0 <= nj < sqrt_P:
            neighbor_rank = ni * sqrt_P + nj + 1  
            neighbors[direction] = neighbor_rank
    return neighbors

def exchange_boundaries(sub_block, neighbors, sub_block_size, comm, num_layers=3):
    direction_tags = {
        'N': 100,
        'NE': 200,
        'E': 300,
        'SE': 400,
        'S': 500,
        'SW': 600,
        'W': 700,
        'NW': 800
    }
    opposites = {
        'N': 'S',
        'NE': 'SW',
        'E': 'W',
        'SE': 'NW',
        'S': 'N',
        'SW': 'NE',
        'W': 'E',
        'NW': 'SE'
    }
    def get_tag(direction, layer):
        return direction_tags[direction] + layer
    send_data = {}
    for direction, neighbor_rank in neighbors.items():
        if neighbor_rank is None:
            send_data[direction] = [['.'] * sub_block_size for _ in range(num_layers)]
            continue  
        if direction in ['N', 'S', 'E', 'W']:
            if direction == 'N':
                lines = [ [cell.name for cell in sub_block[layer]] for layer in range(num_layers) ]
            elif direction == 'S':
                lines = [ [cell.name for cell in sub_block[-(layer + 1)]] for layer in range(num_layers) ]
            elif direction == 'E':
                lines = [ [sub_block[row][-(layer + 1)].name for row in range(sub_block_size)] for layer in range(num_layers) ]
            elif direction == 'W':
                lines = [ [sub_block[row][layer].name for row in range(sub_block_size)] for layer in range(num_layers) ]
            send_data[direction] = lines
        elif direction in ['NE', 'NW', 'SE', 'SW']:
            if direction in ['NE', 'NW']:
                lines = [ [cell.name for cell in sub_block[layer]] for layer in range(num_layers) ]
            elif direction in ['SE', 'SW']:
                lines = [ [cell.name for cell in sub_block[-(layer + 1)]] for layer in range(num_layers) ]
            send_data[direction] = lines
    send_requests = []
    receive_requests = []
    received_data = {dir: [] for dir in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
    for direction, neighbor_rank in neighbors.items():
        if neighbor_rank is None:
            continue
        opposite_dir = opposites[direction]
        for layer in range(num_layers):
            tag = get_tag(opposite_dir, layer)
            recv_req = comm.irecv(source=neighbor_rank, tag=tag)
            receive_requests.append((direction, layer, recv_req))
    for direction, lines in send_data.items():
        neighbor_rank = neighbors[direction]
        for layer in range(num_layers):
            tag = get_tag(direction, layer)
            if layer < len(lines):
                data = lines[layer]
            else:
                data = ['.'] * sub_block_size
            send_req = comm.isend(data, dest=neighbor_rank, tag=tag)
            send_requests.append(send_req)
    MPI.Request.Waitall(send_requests)
    for direction, layer, req in receive_requests:
        data = req.wait()
        received_data[direction].append(data)
    for direction in received_data:
        while len(received_data[direction]) < num_layers:
            received_data[direction].append(['.'] * sub_block_size)
    return received_data

def apply_all_updates(grid, all_updates):
    valid_updates = []
    for update in all_updates:
        if len(update) < 4:
            continue
        preX, preY, newX, newY = update[:4]
        if newX is None or newY is None:
            continue
        if not (0 <= newX < len(grid) and 0 <= newY < len(grid[0])):
            continue
        if grid[preX][preY].name == '.':
            continue
        grid[newX][newY], grid[preX][preY] = grid[preX][preY], grid[newX][newY]
        valid_updates.append(update)
    return valid_updates

#D etermines the rank of the processor that owns the subgrid containing (x, y).
def determine_owner(x, y, sub_block_size, sqrt_P, N):
    if not grid_in_bounds(x, y, N):
        return None  
    sub_block_row = x // sub_block_size
    sub_block_col = y // sub_block_size
    if sub_block_row >= sqrt_P or sub_block_col >= sqrt_P:
        return None  
    owner_rank = sub_block_row * sqrt_P + sub_block_col + 1  
    return owner_rank

#Continuously listens for RPC requests and processes them.
def rpc_request_handler(shared_data, rank, sub_block_size, sqrt_P, N, comm):
    while True:
        if comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_RPC_REQUEST):
            request = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RPC_REQUEST)
            requester_rank, tx, ty = request  
            with shared_data.lock:
                local_sub_block = [row.copy() for row in shared_data.sub_block]
                local_received_boundaries = {k: [line.copy() for line in v] for k, v in shared_data.received_boundaries.items()}
            target_count = count_number_of_targets(tx, ty, rank, local_sub_block, local_received_boundaries, N, sub_block_size, sqrt_P, comm)
            comm.send(target_count, dest=requester_rank, tag=TAG_RPC_RESPONSE)
        else:
            time.sleep(0.01)

def movement_phase(rank, sub_block, received_boundaries, sub_block_size, sqrt_P, N, comm):
    updates = []
    for x in range(sub_block_size):
        for y in range(sub_block_size):
            if sub_block[x][y].name != 'A':
                continue  
            global_x, global_y = get_global_idx(x, y, rank, sub_block_size, sqrt_P, N)
            res = move_air_unit(global_x, global_y, rank, sub_block, received_boundaries, N, sub_block_size, sqrt_P, comm)
            if res is not None and res != (None, None):
                new_x, new_y = res
                updates.append([global_x, global_y, new_x, new_y, rank])
    return updates

def move_air_unit(x, y, rank, sub_block, received_boundaries, N, sub_block_size, sqrt_P, comm):
    possible_moves = []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    local_x = x % sub_block_size
    local_y = y % sub_block_size
    current_targets = count_number_of_targets(x, y, rank, sub_block, received_boundaries, N, sub_block_size, sqrt_P, comm)
    at_corner = (local_x == 0 or local_x == sub_block_size -1) and (local_y == 0 or local_y == sub_block_size -1)
    if at_corner:
        if local_x == 0 and local_y == 0:
            corner = 'NW'  
        elif local_x == 0 and local_y == sub_block_size - 1:
            corner = 'NE'  
        elif local_x == sub_block_size - 1 and local_y == 0:
            corner = 'SW'  
        elif local_x == sub_block_size - 1 and local_y == sub_block_size - 1:
            corner = 'SE'  
        else:
            corner = None  
    else:
        corner = None  
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < N and 0 <= ny < N):
            continue
        same_subblock = (nx // sub_block_size) == ((rank - 1) // sqrt_P) and \
                        (ny // sub_block_size) == ((rank - 1) % sqrt_P)
        if same_subblock:
            local_x, local_y = nx % sub_block_size, ny % sub_block_size
            if sub_block[local_x][local_y].name != '.':
                continue  
        else:
            direction = get_direction(x, y, nx, ny, sqrt_P, sub_block_size)
            if not direction:
                continue  
            primary_direction = direction
            boundary_index_offset = 0
            is_diagonal_move = abs(dx) == 1 and abs(dy) ==1
            if is_diagonal_move and at_corner and direction == corner:
                primary_direction = direction  
            elif is_diagonal_move :
                if direction in ['NE', 'NW']:
                    primary_direction = 'N'
                elif direction in ['SE', 'SW']:
                    primary_direction = 'S'
            else:
                primary_direction = direction
            boundary_layers = received_boundaries.get(primary_direction, [])
            cell = '.'
            if primary_direction in ['N', 'S']:
                layer = 0  
                if len(boundary_layers) > layer:
                    boundary = boundary_layers[layer]
                    boundary_index = ny % sub_block_size
                    if 0 <= boundary_index < len(boundary):
                        cell = boundary[boundary_index]
            elif primary_direction in ['E', 'W']:
                layer = 0  
                if len(boundary_layers) > layer:
                    boundary = boundary_layers[layer]
                    boundary_index = nx % sub_block_size
                    if 0 <= boundary_index < len(boundary):
                        cell = boundary[boundary_index]
            elif primary_direction in ['NE', 'SE']:
                if len(boundary_layers) > 0 and len(boundary_layers[0]) > 0:
                    cell = boundary_layers[0][0]  
            elif primary_direction in ['NW', 'SW']:
                if len(boundary_layers) > 0 and len(boundary_layers[0]) > 0:
                    cell = boundary_layers[0][sub_block_size-1]
            else:
                cell = '.'  
            if cell != '.':
                continue
        targets = count_number_of_targets(nx, ny, rank, sub_block, received_boundaries, N, sub_block_size, sqrt_P, comm)
        possible_moves.append((nx, ny, targets))
    if not possible_moves:
        return (None, None)  
    possible_moves.sort(key=lambda move: (-move[2], move[0], move[1]))
    best_move = possible_moves[0]
    best_targets = best_move[2]
    if best_targets > current_targets:
        return (best_move[0], best_move[1])
    else:
        return (None, None)  
def target_direction(boundary_x, boundary_y, sub_block_size):
    if boundary_x < 0 and boundary_y < 0:
        target_boundary = 'NW'  
    elif boundary_x >= sub_block_size and boundary_y < 0:
        target_boundary = 'SW'  
    elif boundary_x < 0 and boundary_y >= sub_block_size:
        target_boundary = 'NE'  
    elif boundary_x >= sub_block_size and boundary_y >= sub_block_size:
        target_boundary = 'SE'  
    elif (0<=boundary_x<sub_block_size) and boundary_y>= sub_block_size:
        target_boundary = 'E'
    elif boundary_x>= sub_block_size and (0<=boundary_y<sub_block_size):
        target_boundary = 'S'
    elif (0<=boundary_x<sub_block_size) and boundary_y < 0:
        target_boundary = 'W'
    elif boundary_x < 0 and (0<=boundary_y<sub_block_size):
        target_boundary = 'N'
    else:
        target_boundary = None  
    return target_boundary
def get_direction(x, y, nx, ny, sqrt_P, sub_block_size):
    dx = nx - x
    dy = ny - y
    direction_map = {
        (-1, -1): 'NW',
        (-1, 0): 'N',
        (-1, 1): 'NE',
        (0, -1): 'W',
        (0, 1): 'E',
        (1, -1): 'SW',
        (1, 0): 'S',
        (1, 1): 'SE',
        (-2, -2): 'NW',  
        (-2, -1): 'N',
        (-2, 0): 'N',
        (-2, 1): 'NE',
        (-2, 2): 'NE',
        (-1, -2): 'NW',
        (-1, 2): 'NE',
        (1, -2): 'SW',
        (1, 2): 'SE',
        (2, -2): 'SW',
        (2, -1): 'S',
        (2, 0): 'S',
        (2, 1): 'SE',
        (2, 2): 'SE'
    }
    direction_map = {
    (-1, -1): 'NW',  
    (-1,  0): 'N',   
    (-1,  1): 'NE',  
    ( 0, -1): 'W',   
    ( 0,  1): 'E',   
    ( 1, -1): 'SW',  
    ( 1,  0): 'S',   
    ( 1,  1): 'SE',  
    (-2, -2): 'NW',  
    (-2, -1): 'N',   
    (-2,  0): 'N',   
    (-2,  1): 'NE',  
    (-2,  2): 'NE',  
    (-1, -2): 'NW',  
    (-1,  2): 'NE',  
    ( 0, -2): 'W',   
    ( 0,  2): 'E',   
    ( 1, -2): 'SW',  
    ( 1,  2): 'SE',  
    ( 2, -2): 'SW',  
    ( 2, -1): 'S',   
    ( 2,  0): 'S',   
    ( 2,  1): 'SE',  
    ( 2,  2): 'SE',  
}
    return direction_map.get((dx, dy), None)

def count_number_of_targets(x, y, rank, sub_block, received_boundaries, N, sub_block_size, sqrt_P, comm):
    total_targets = 0
    counted_directions = set()
    attack_pattern = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    local_x, local_y = x % sub_block_size, y % sub_block_size
    owner_rank = determine_owner(x, y, sub_block_size, sqrt_P, N)
    if owner_rank is None:
        print("Something goes very wrong")
    elif owner_rank != rank:
        comm.send([rank, x, y], dest=owner_rank, tag=TAG_RPC_REQUEST)
        target_count = comm.recv(source=owner_rank, tag=TAG_RPC_RESPONSE)
        total_targets += target_count
        return total_targets
    for dx, dy in attack_pattern:
        tx, ty = x + dx, y + dy
        boundary_x,boundary_y = local_x+dx, local_y + dy
        target_boundary = target_direction(boundary_x,boundary_y, sub_block_size)
        if not grid_in_bounds(tx, ty, N):
            continue
        direction = get_direction(x, y, tx, ty, sqrt_P,sub_block_size)
        if direction in counted_directions:
            continue
        if not direction:
            continue  
        same_subblock = (tx // sub_block_size) == ((rank - 1) // sqrt_P) and \
                        (ty // sub_block_size) == ((rank - 1) % sqrt_P)
        if same_subblock:
            local_nx, local_ny = tx % sub_block_size, ty % sub_block_size
            target_unit = sub_block[local_nx][local_ny]
        else:
            boundary_lines = received_boundaries.get(target_boundary, [])
            if target_boundary in ['N', 'S', 'E', 'W']:
                if target_boundary == 'N':
                    if 0 <= ty % sub_block_size < len(boundary_lines):
                        cell = boundary_lines[0][ty % sub_block_size]
                    else:
                        cell = '.'
                elif target_boundary == 'S':
                    if 0 <= ty % sub_block_size < len(boundary_lines):
                        cell = boundary_lines[0][ty % sub_block_size]
                    else:
                        cell = '.'
                elif target_boundary == 'E':
                    if 0 <= tx % sub_block_size < len(boundary_lines):
                        cell = boundary_lines[0][tx % sub_block_size]
                    else:
                        cell = '.'
                elif target_boundary == 'W':
                    if 0 <= tx % sub_block_size < len(boundary_lines):
                        cell = boundary_lines[0][tx % sub_block_size]
                    else:
                        cell = '.'
                else:
                    cell = '.'
                target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
            else:
                if target_boundary in ['NE', 'SE']:
                    cell = boundary_lines[0][0] if boundary_lines and boundary_lines[0] else '.'
                elif target_boundary in ['NW', 'SW']:
                    cell = boundary_lines[0][-1] if boundary_lines and boundary_lines[0] else '.'
                else:
                    cell = '.'
                target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
        if target_unit.name != '.' and target_unit.name != 'A':
            total_targets += 1
            counted_directions.add(direction)
        if target_unit.name =="A":
            continue
        if target_unit.name == '.':
            far_x, far_y = x + 2 * dx, y + 2 * dy
            boundary_x_far,boundary_y_far = local_x+2*dx, local_y + 2*dy
            target_boundary_far = target_direction(boundary_x_far,boundary_y_far,sub_block_size)
            if not grid_in_bounds(far_x, far_y, N):
                continue
            direction_far = get_direction(x, y, far_x, far_y, sqrt_P,sub_block_size)
            if direction_far in counted_directions:
                continue
            if not direction_far:
                continue 
            same_subblock_far = (far_x // sub_block_size) == ((rank - 1) // sqrt_P) and \
                                (far_y // sub_block_size) == ((rank - 1) % sqrt_P)
            if same_subblock_far:
                local_fx, local_fy = far_x % sub_block_size, far_y % sub_block_size
                far_unit = sub_block[local_fx][local_fy]
            else:
                cell_far = '.'
                boundary_lines_far = received_boundaries.get(target_boundary_far, [])
                if target_boundary_far in ['N', 'S', 'E', 'W']:
                    if target_boundary_far == 'N':
                        if 0 <= far_y % sub_block_size < len(boundary_lines_far):
                            cell_far = boundary_lines_far[sub_block_size-(far_x%sub_block_size)-1][(far_y + 2*dy) % sub_block_size]
                        else:
                            cell_far = '.'
                    elif target_boundary_far == 'S':
                        if 0 <= far_y % sub_block_size < len(boundary_lines_far):
                            cell_far = boundary_lines_far[far_x%sub_block_size][(far_y + 2*dy) % sub_block_size] 
                        else:
                            cell_far = '.'
                    elif target_boundary_far == 'E':
                        if 0 <= far_x % sub_block_size < len(boundary_lines_far):
                            cell_far = boundary_lines_far[far_y%sub_block_size][(far_x + 2*dx) % sub_block_size]
                        else:
                            cell_far = '.'
                    elif target_boundary_far == 'W':
                        if 0 <= far_x % sub_block_size < len(boundary_lines_far):
                            cell_far = boundary_lines_far[sub_block_size-(far_y%sub_block_size)-1][(far_x + 2*dx) % sub_block_size]
                        else:
                            cell_far = '.'
                    else:
                        cell_far = '.'
                    far_unit = UNIT(cell_far, 0, 0) if cell_far == '.' else UNIT(cell_far, faction_stats[cell_far]['max_health'], faction_stats[cell_far]['attack_power'])
                else:
                    if target_boundary_far == "NE":
                        cell_far = boundary_lines_far[1-local_x][1-(sub_block_size-1-local_y)] if boundary_lines_far and boundary_lines_far[0] else '.'
                    elif target_boundary_far == "SE":
                        cell_far = boundary_lines_far[1-(sub_block_size-1-local_x)][1-(sub_block_size-1-local_y)] if boundary_lines_far and boundary_lines_far[0] else '.'
                    elif target_boundary_far == "NW":
                        cell_far = boundary_lines_far[1-local_x][-2+local_y] if boundary_lines_far and boundary_lines_far[0] else '.'
                    elif target_boundary_far == "SW":
                        cell_far = boundary_lines_far[1-(sub_block_size-1-local_x)][-2+local_y] if boundary_lines_far and boundary_lines_far[0] else '.'
                    else:
                        cell = '.'
                    far_unit = UNIT(cell_far, 0, 0) if cell_far == '.' else UNIT(cell_far, faction_stats[cell_far]['max_health'], faction_stats[cell_far]['attack_power'])
            if far_unit.name != '.' and far_unit.name != 'A':
                total_targets += 1
                counted_directions.add(direction_far)
            if far_unit.name =="A":
                continue
    return total_targets

# Checks if the given coordinates are within the grid bounds.
def grid_in_bounds(x, y, N):
    return 0 <= x < N and 0 <= y < N

def attack_phase_worker(rank, sub_block, received_boundaries, sub_block_size, sqrt_P, N):
    attacks = []
    heals = []
    for x in range(sub_block_size):
        for y in range(sub_block_size):
            unit = sub_block[x][y]
            if unit.name == '.':
                continue  
            global_x, global_y = get_global_idx(x, y, rank, sub_block_size, sqrt_P, N)
            max_health = faction_stats[unit.name]['max_health']
            if unit.hp >= (0.5 * max_health):
                target_positions = []
                if unit.name == 'A':
                    target_positions = determine_air_unit_targets_worker(global_x, global_y, rank, sub_block, received_boundaries, sub_block_size, sqrt_P, N)
                else:
                    target_positions = determine_standard_unit_targets_worker(global_x, global_y, unit, rank, sub_block, received_boundaries, sub_block_size, sqrt_P, N)
                for tx, ty in target_positions:
                    if grid_in_bounds(tx, ty, N):
                        damage = unit.attack_power
                        attacks.append([global_x, global_y, tx, ty, damage])
            else:
                heal_amount = faction_stats[unit.name]['heal_rate']
                heals.append([global_x, global_y, heal_amount])
    return attacks, heals

def determine_air_unit_targets_worker(global_x, global_y, rank, sub_block, received_boundaries, sub_block_size, sqrt_P, N):
    target_positions = []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    local_x, local_y = global_x % sub_block_size, global_y % sub_block_size
    for dx, dy in directions:
        adj_x, adj_y = global_x + dx, global_y + dy
        boundary_x,boundary_y = local_x+dx, local_y + dy
        target_boundary = target_direction(boundary_x,boundary_y, sub_block_size)
        if not grid_in_bounds(adj_x, adj_y, N):
            continue
        same_subblock = (adj_x // sub_block_size) == ((rank - 1) // sqrt_P) and \
                        (adj_y // sub_block_size) == ((rank - 1) % sqrt_P)
        if same_subblock:
            local_nx, local_ny = adj_x % sub_block_size, adj_y % sub_block_size
            target_unit = sub_block[local_nx][local_ny]
        else:
            direction = get_direction(global_x, global_y, adj_x, adj_y, sqrt_P, sub_block_size)
            if not direction:
                continue  
            boundary_lines = received_boundaries.get(target_boundary, [])
            if not boundary_lines:
                target_unit = UNIT('.', 0, 0)  
            else:
                if target_boundary in ['N', 'S']:
                    boundary_index = adj_y % sub_block_size
                    if 0 <= boundary_index < len(boundary_lines[0]):
                        cell = boundary_lines[0][boundary_index]
                        target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
                    else:
                        target_unit = UNIT('.', 0, 0)
                elif target_boundary in ['E', 'W']:
                    boundary_index = adj_x % sub_block_size
                    if 0 <= boundary_index < len(boundary_lines[0]):
                        cell = boundary_lines[0][boundary_index]
                        target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
                    else:
                        target_unit = UNIT('.', 0, 0)
                elif target_boundary in ['NE', 'SE', 'NW', 'SW']:
                    if target_boundary in ['NE', 'SE']:
                        cell = boundary_lines[0][0] if boundary_lines and boundary_lines[0] else '.'
                    elif target_boundary in ['NW', 'SW']:
                        cell = boundary_lines[0][-1] if boundary_lines and boundary_lines[0] else '.'
                    else:
                        cell = '.'  
                    target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
                else:
                    target_unit = UNIT('.', 0, 0)
        if target_unit.name != '.' and target_unit.name != 'A':
            target_positions.append((adj_x, adj_y))
        if target_unit.name == '.':
            far_x, far_y = global_x + 2 * dx, global_y + 2 * dy
            boundary_x_far,boundary_y_far = local_x+2*dx, local_y + 2*dy
            target_boundary_far = target_direction(boundary_x_far,boundary_y_far,sub_block_size)
            if not grid_in_bounds(far_x, far_y, N):
                continue
            same_subblock_far = (far_x // sub_block_size) == ((rank - 1) // sqrt_P) and \
                                (far_y // sub_block_size) == ((rank - 1) % sqrt_P)
            if same_subblock_far:
                local_fx, local_fy = far_x % sub_block_size, far_y % sub_block_size
                far_unit = sub_block[local_fx][local_fy]
            else:
                direction_far = get_direction(global_x, global_y, far_x, far_y, sqrt_P, sub_block_size)
                if not direction_far:
                    far_unit = UNIT('.', 0, 0)
                else:
                    boundary_lines_far = received_boundaries.get(target_boundary_far, [])
                    if not boundary_lines_far:
                        far_unit = UNIT('.', 0, 0)
                    else:
                        if target_boundary_far == 'N':
                            boundary_index_far = far_y % sub_block_size
                            if 0 <= boundary_index_far < len(boundary_lines_far[0]):
                                cell_far = boundary_lines_far[sub_block_size-(far_x%sub_block_size)-1][(far_y + 2*dy) % sub_block_size]
                                far_unit = UNIT(cell_far, 0, 0) if cell_far == '.' else UNIT(cell_far, faction_stats[cell_far]['max_health'], faction_stats[cell_far]['attack_power'])
                            else:
                                far_unit = UNIT('.', 0, 0)
                        elif target_boundary_far == 'S':
                            boundary_index_far = far_y % sub_block_size
                            if 0 <= boundary_index_far < len(boundary_lines_far[0]):
                                cell_far = boundary_lines_far[far_x%sub_block_size][(far_y + 2*dy) % sub_block_size] 
                                far_unit = UNIT(cell_far, 0, 0) if cell_far == '.' else UNIT(cell_far, faction_stats[cell_far]['max_health'], faction_stats[cell_far]['attack_power'])
                            else:
                                far_unit = UNIT('.', 0, 0)
                        elif target_boundary_far == 'E':
                            boundary_index_far = far_x % sub_block_size
                            if 0 <= boundary_index_far < len(boundary_lines_far[0]):
                                cell_far = boundary_lines_far[far_y%sub_block_size][(far_x + 2*dx) % sub_block_size]
                                far_unit = UNIT(cell_far, 0, 0) if cell_far == '.' else UNIT(cell_far, faction_stats[cell_far]['max_health'], faction_stats[cell_far]['attack_power'])
                            else:
                                far_unit = UNIT('.', 0, 0)
                        elif target_boundary_far == 'W':
                            boundary_index_far = far_x % sub_block_size
                            if 0 <= boundary_index_far < len(boundary_lines_far[0]):
                                cell_far = boundary_lines_far[sub_block_size-(far_y%sub_block_size)-1][(far_x + 2*dx) % sub_block_size]
                                far_unit = UNIT(cell_far, 0, 0) if cell_far == '.' else UNIT(cell_far, faction_stats[cell_far]['max_health'], faction_stats[cell_far]['attack_power'])
                            else:
                                far_unit = UNIT('.', 0, 0)
                        elif target_boundary_far in ['NE', 'NW', 'SE', 'SW']:
                            if target_boundary_far == "NE":
                                ell_far = boundary_lines_far[1-local_x][1-(sub_block_size-1-local_y)] if boundary_lines_far and boundary_lines_far[0] else '.'
                            elif target_boundary_far == "SE":
                                cell_far = boundary_lines_far[1-(sub_block_size-1-local_x)][1-(sub_block_size-1-local_y)] if boundary_lines_far and boundary_lines_far[0] else '.'
                            elif target_boundary_far == "NW":
                                cell_far = boundary_lines_far[1-local_x][-2+local_y] if boundary_lines_far and boundary_lines_far[0] else '.'
                            elif target_boundary_far == "SW":
                                cell_far = boundary_lines_far[1-(sub_block_size-1-local_x)][-2+local_y] if boundary_lines_far and boundary_lines_far[0] else '.'   
                            else:
                                cell_far = '.'  
                            far_unit = UNIT(cell_far, 0, 0) if cell_far == '.' else UNIT(cell_far, faction_stats[cell_far]['max_health'], faction_stats[cell_far]['attack_power'])
                        else:
                            far_unit = UNIT('.', 0, 0)
            if far_unit.name != '.' and far_unit.name != 'A':
                target_positions.append((far_x, far_y))
    return target_positions

def determine_standard_unit_targets_worker(global_x, global_y, unit, rank, sub_block, received_boundaries, sub_block_size, sqrt_P, N):
    target_positions = []
    local_x, local_y = global_x % sub_block_size, global_y % sub_block_size
    at_corner = (local_x == 0 or local_x == sub_block_size -1) and (local_y == 0 or local_y == sub_block_size -1)
    if at_corner:
        if local_x == 0 and local_y == 0:
            corner = 'NW'  
        elif local_x == 0 and local_y == sub_block_size - 1:
            corner = 'NE'  
        elif local_x == sub_block_size - 1 and local_y == 0:
            corner = 'SW'  
        elif local_x == sub_block_size - 1 and local_y == sub_block_size - 1:
            corner = 'SE'  
        else:
            corner = None  
    else:
        corner = None  
    for dx, dy in faction_stats[unit.name]['attack_pattern']:
        tx, ty = global_x + dx, global_y + dy
        if not grid_in_bounds(tx, ty, N):
            continue
        same_subblock = (tx // sub_block_size) == ((rank - 1) // sqrt_P) and \
                        (ty // sub_block_size) == ((rank - 1) % sqrt_P)
        if same_subblock:
            local_tx, local_ty = tx % sub_block_size, ty % sub_block_size
            target_unit = sub_block[local_tx][local_ty]
        else:
            direction = get_direction(global_x, global_y, tx, ty, sqrt_P, sub_block_size)
            if not direction:
                continue  
            primary_direction = direction
            boundary_index_offset = 0
            is_diagonal_attack = abs(dx) == 1 and abs(dy) ==1
            if is_diagonal_attack and at_corner and direction == corner:
                primary_direction = direction  
            elif is_diagonal_attack :
                if corner == "NE":
                    if direction == "SE":
                        primary_direction = 'E'
                    elif direction == "NW":
                        primary_direction = 'N'
                elif corner == "NW":
                    if direction == "SW":
                        primary_direction = 'W'
                    elif direction == "NE":
                        primary_direction = 'N'
                elif corner == "SE":
                    if direction == "SW":
                        primary_direction = 'S'
                    elif direction == "NE":
                        primary_direction = 'E'
                elif corner == "SW":
                    if direction == "SE":
                        primary_direction = 'S'
                    elif direction == "NW":
                        primary_direction = 'W'
            else:
                primary_direction = direction
            boundary_layers = received_boundaries.get(primary_direction, [])
            if not boundary_layers:
                target_unit = UNIT('.', 0, 0)  
            else:
                if primary_direction in ['N', 'S']:
                    boundary_index = ty % sub_block_size
                    if 0 <= boundary_index < len(boundary_layers[0]):
                        cell = boundary_layers[0][boundary_index]
                        target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
                    else:
                        target_unit = UNIT('.', 0, 0)
                elif primary_direction in ['E', 'W']:
                    boundary_index = tx % sub_block_size
                    if 0 <= boundary_index < len(boundary_layers[0]):
                        cell = boundary_layers[0][boundary_index]
                        target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
                    else:
                        target_unit = UNIT('.', 0, 0)
                elif primary_direction in ['NE', 'NW', 'SE', 'SW']:
                    if primary_direction in ['NE', 'SE']:
                        cell = boundary_layers[0][0] if boundary_layers and boundary_layers[0] else '.'
                    elif primary_direction in ['NW', 'SW']:
                        cell = boundary_layers[0][-1] if boundary_layers and boundary_layers[0] else '.'
                    else:
                        cell = '.'  
                    target_unit = UNIT(cell, 0, 0) if cell == '.' else UNIT(cell, faction_stats[cell]['max_health'], faction_stats[cell]['attack_power'])
                else:
                    target_unit = UNIT('.', 0, 0)
        if target_unit.name != '.' and target_unit.name != unit.name:
            target_positions.append((tx, ty))
    return target_positions

def apply_attacks_master(attacks, grid, N):
    damage_accumulator = {}
    record = []
    attackers = set()
    for attack in attacks:
        if len(attack) != 5:
            continue  
        preX, preY, tx, ty, damage = attack
        if not grid_in_bounds(preX, preY, N) or not grid_in_bounds(tx, ty, N):
            continue  
        attacker = grid[preX][preY]
        target = grid[tx][ty]
        if attacker.name == '.' or target.name == '.' or attacker.name == target.name:
            continue  
        adjusted_damage = damage
        target_key = (tx, ty)
        if target_key in damage_accumulator:
            damage_accumulator[target_key] += adjusted_damage
        else:
            damage_accumulator[target_key] = adjusted_damage
        record.append([preX, preY, tx, ty, adjusted_damage])
        attackers.add((preX, preY))
    for (tx, ty), total_damage in damage_accumulator.items():
        target_unit = grid[tx][ty]
        if target_unit.name == 'E':
            total_damage = math.floor(total_damage / 2)
        else:
            total_damage = total_damage
        target_unit.hp -= total_damage
    return record, attackers

def check_fire_units(attacks, grid, N):
    increase_attack_power = {}
    for attack in attacks:
        if len(attack) != 5:
            continue  
        preX, preY, tx, ty, damage = attack
        attacker = grid[preX][preY]
        target = grid[tx][ty]
        if attacker.name != 'F':
            continue
        if target.hp <= 0:
            if (preX, preY) not in increase_attack_power:
                increase_attack_power[(preX, preY)] = 1
            else:
                increase_attack_power[(preX, preY)] += 1
    for (x, y), count in increase_attack_power.items():
        attacker = grid[x][y]
        attacker.attack_power += count
        if attacker.attack_power > faction_stats['F']['max_attack_power']:
            attacker.attack_power = faction_stats['F']['max_attack_power']

def select_cell_to_flood(grid, N, x, y):
    candidates = []
    directions = [ (0,1),(0,-1) ,(-1,0), (1,0), (-1,-1), (-1,1), (1,1), (1,-1) ]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < N and 0 <= ny < N and grid[nx][ny].name == '.':
            candidates.append( (nx, ny) )
    candidates.sort(key=lambda pos: (pos[0], pos[1]))
    return candidates[0] if candidates else (None, None)

def flood_one_cell(grid, N):
    for x in range(N):
        for y in range(N):
            unit = grid[x][y]
            if unit.name != 'W':
                continue
            X, Y = select_cell_to_flood(grid, N, x, y)
            if X is not None:
                grid[X][Y] = UNIT('W', faction_stats['W']['max_health'], faction_stats['W']['attack_power'])

def remove_dead_units(grid, N):
    for x in range(N):
        for y in range(N):
            unit = grid[x][y]
            if unit.hp <= 0:
                grid[x][y] = UNIT('.', 0, 0)

def heal_units(to_heal, N, grid):
    for x, y in to_heal:
        unit = grid[x][y]
        previous_hp = unit.hp
        if unit.name != '.' and unit.hp > 0:
            grid[x][y].hp = min(unit.hp + faction_stats[unit.name]['heal_rate'],
                                faction_stats[unit.name]['max_health'])
            #print(f"\tUnit {unit.name} at ({x}, {y}) healed from {previous_hp} to {unit.hp} HP.")

def nullify_attack_power(grid, N):
    for x in range(N):
        for y in range(N):
            unit = grid[x][y]
            if unit.name == 'F':
                grid[x][y].attack_power = faction_stats['F']['attack_power']


# Master Process
if rank == 0:
    if len(sys.argv) < 3:
        print("Usage: mpiexec -n <P> python script.py <input_file> <output_file>")
        sys.exit()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    N, W, T, R, waves = parse_input(input_file)
    grid = [[UNIT('.', 0, 0) for _ in range(N)] for _ in range(N)]
    blocks = split_grid(grid, num_processors)
    sqrt_P = int(math.sqrt(num_processors - 1))
    sub_block_size = N // sqrt_P
    for worker_rank in range(1, num_processors):
        sub_block, position = blocks[worker_rank - 1]
        comm.send( (sub_block, position), dest=worker_rank, tag=TAG_INIT )
    for w in range(1, W + 1):
        #print(f"\n=== WAVE {w} ===")
        wave_units = waves[w]
        populate_grid(grid, N, wave_units)
        #print("Grid after initialization:")
        #visualise(grid)
        for r in range(1, R + 1):
            #print(f"\n--- ROUND {r} ---")
            #print("Movement Phase:")
            for worker_rank in range(1, num_processors):
                sub_block = [row.copy() for row in split_grid(grid, num_processors)[worker_rank -1][0]]
                comm.send(sub_block, dest=worker_rank, tag=TAG_MOVEMENT_PHASE)
            all_updates = []
            for worker_rank in range(1, num_processors):
                updates = comm.recv(source=worker_rank, tag=TAG_MOVEMENT_PHASE)
                all_updates.extend(updates)
            valid_updates = apply_all_updates(grid, all_updates)
            if valid_updates:
                for preX, preY, newX, newY, _ in valid_updates:
                    #print(f"\tUnit {grid[newX][newY].name} moved from ({preX}, {preY}) to ({newX}, {newY}).")
                    pass
            else:
                #print("\tNo air unit movements.")
                pass
            #print("Grid after movement:")
            #visualise(grid)
            #print("Attack Phase:")
            for worker_rank in range(1, num_processors):
                sub_block = [row.copy() for row in split_grid(grid, num_processors)[worker_rank -1][0]]
                comm.send(sub_block, dest=worker_rank, tag=TAG_ATTACK_PHASE)
            all_attacks = []
            all_heals = []
            for worker_rank in range(1, num_processors):
                attack_data, heal_data = comm.recv(source=worker_rank, tag=TAG_ATTACK_PHASE)
                all_attacks.extend(attack_data)
                all_heals.extend(heal_data)
            record, attackers = apply_attacks_master(all_attacks, grid, N)
            check_fire_units(all_attacks, grid, N)
            if record:
                for x, y, tx, ty, damage in record:
                    attacker = grid[x][y].name
                    #print(f"\tUnit {attacker} at ({x}, {y}) attacked unit at ({tx}, {ty}) with damage: {damage}.")
            else:
                #print("\tNo attacks this round.")
                pass
            #print("\nHealth Reductions:")
            for x, y, tx, ty, damage in record:
                if grid[tx][ty].hp <= 0:
                    #print(f"\tUnit at ({tx}, {ty}) died.")
                    pass
                else:
                    #print(f"\tUnit at ({tx}, {ty}) now has {grid[tx][ty].hp} HP.")
                    pass
            remove_dead_units(grid, N)
            #print("\nHealing Phase:")
            additional_heals = []
            for i in range(N):
                for j in range(N):
                    unit = grid[i][j]
                    if (unit.name != '.' and 
                        unit.hp < faction_stats[unit.name]['max_health'] and 
                        (i, j) not in attackers):
                        additional_heals.append((i, j))
            if additional_heals:
                heal_units(additional_heals,N,grid)
            else:
                #print("\tNo units healed this round.")
                pass
            #print("\nGrid after this round:")
            #visualise(grid)
        #print(f"\nEnd of Wave {w}:")
        flood_one_cell(grid, N)
        nullify_attack_power(grid, N)
        #print("Grid after end-of-wave actions:")
        #visualise(grid)
    #print("\nAll waves completed.")
    with open(output_file, 'w') as f:
        for row in grid:
            line = " ".join([cell.name for cell in row])
            f.write(line + "\n")
    for worker_rank in range(1, num_processors):
        comm.send(None, dest=worker_rank, tag=TAG_TERMINATE)

# Worker Processes
else:
    received = comm.recv(source=0, tag=TAG_INIT)
    if received is None:
        sys.exit()
    sub_block, position = received
    sqrt_P = int(math.sqrt(num_processors - 1))
    sub_block_size = len(sub_block)
    N = sub_block_size * sqrt_P
    neighbors = get_neighbors(rank, sqrt_P)
    initial_boundaries = {}
    shared_data = SharedData([row.copy() for row in sub_block], initial_boundaries)
    rpc_thread = threading.Thread(
        target=rpc_request_handler,
        args=(shared_data, rank, sub_block_size, sqrt_P, N, comm)
    )
    rpc_thread.daemon = True  
    rpc_thread.start()  
    while True:
        status = MPI.Status()
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == TAG_MOVEMENT_PHASE:
            current_sub_block_new = comm.recv(source=0, tag=TAG_MOVEMENT_PHASE)
            if current_sub_block_new is None:
                updates = []
                comm.send(updates, dest=0, tag=TAG_MOVEMENT_PHASE)
                continue
            received_boundaries_new = exchange_boundaries(current_sub_block_new, neighbors, sub_block_size, comm, num_layers=NUM_LAYERS)
            with shared_data.lock:
                shared_data.sub_block.clear()
                shared_data.sub_block.extend([row.copy() for row in current_sub_block_new])
                shared_data.received_boundaries = received_boundaries_new
            updates = movement_phase(rank, shared_data.sub_block, shared_data.received_boundaries, sub_block_size, sqrt_P, N, comm)
            comm.send(updates, dest=0, tag=TAG_MOVEMENT_PHASE)
        elif tag == TAG_ATTACK_PHASE:
            current_sub_block_new = comm.recv(source=0, tag=TAG_ATTACK_PHASE)
            if current_sub_block_new is None:
                attack_data = []
                comm.send(attack_data, dest=0, tag=TAG_ATTACK_PHASE)
                continue
            received_boundaries_new = exchange_boundaries(current_sub_block_new, neighbors, sub_block_size, comm, num_layers=NUM_LAYERS)
            with shared_data.lock:
                shared_data.sub_block.clear()
                shared_data.sub_block.extend([row.copy() for row in current_sub_block_new])
                shared_data.received_boundaries = received_boundaries_new
            attack_data, heal_data = attack_phase_worker(rank, shared_data.sub_block, shared_data.received_boundaries, sub_block_size, sqrt_P, N)
            comm.send((attack_data, heal_data), dest=0, tag=TAG_ATTACK_PHASE)
        elif tag == TAG_TERMINATE:
            comm.recv(source=0, tag=TAG_TERMINATE)  
            break
        else:
            comm.recv(source=0, tag=tag)
            #print(f"Worker {rank}: Received unknown tag {tag}.")
            break
