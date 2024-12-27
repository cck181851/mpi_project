from mpi4py import MPI
import math

class UNIT:
    def __init__(self,name,hp,attack_power):
        self.name=name
        self.hp=hp
        self.attack_power=attack_power

faction_stats = {
    'E': {'heal_rate':3,'attack_power': 2, 'max_health': 18, 'attack_pattern': [(-1, 0), (1, 0), (0, -1), (0, 1)]},
    'F': {'heal_rate':1,'attack_power': 4, 'max_health': 12, 'max_attack_power': 6, 'attack_pattern': [(-1, -1), (-1, 0), (-1, 1),
                                                                                                       (0, -1),         (0, 1),
                                                                                                       (1, -1), (1, 0), (1, 1)]},
    'W': {'heal_rate':2,'attack_power': 3, 'max_health': 14, 'attack_pattern': [(-1, -1), (-1, 1), (1, -1), (1, 1)]},
    'A': {'heal_rate':2,'attack_power': 2, 'max_health': 10, 'attack_pattern': [(-1, -1), (-1, 0), (-1, 1),  
                                                                                 (0, -1),         (0, 1),
                                                                                 (1, -1), (1, 0), (1, 1)]}
}

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
    coords = coord_line.split(',')
    return [tuple(map(int, coord.split())) for coord in coords]

def populate_grid(grid, N, wave_units):    
    for faction, positions in wave_units.items():
        for x, y in positions:
            unit=UNIT(faction,faction_stats[faction]['max_health'],faction_stats[faction]['attack_power'])
            if grid[x][y].name=='.':
                grid[x][y] = unit         


def split_grid(grid, P):
    N = len(grid)
    sqrt_P = int((P-1) ** 0.5) 
    block_size = N // sqrt_P

    blocks = []  

    for i in range(sqrt_P):
        for j in range(sqrt_P):
            block = [row[j*block_size:(j+1)*block_size] for row in grid[i*block_size:(i+1)*block_size]]
            blocks.append((block, (i, j)))

    return blocks

def receive_neighbors(sub_block, num_processors, rank):
    send_requests = []
    recv_requests = []

    for i in range(1, num_processors):
        if i != rank:
            # Non-blocking send
            send_requests.append(comm.isend(sub_block, dest=i, tag=100 + i))
            
            # Non-blocking receive
            recv_requests.append((comm.irecv(source=i, tag=100 + rank), i))

    # Wait for all send requests to complete
    MPI.Request.Waitall(send_requests)

    # Wait for all receive requests and collect data
    received_blocks = []
    for req, i in recv_requests:
        received_blocks.append([req.wait(), i])

    return received_blocks

def movement_phase(rank, sub_block, neighbors, N, sub_block_size):
    updates=[]
    for x in range(sub_block_size):
        for y in range(sub_block_size):
            if sub_block[x][y].name!='A': continue             
            X,Y=get_global_idx(x,y,rank,sub_block_size,N)       
            res= move_air_unit(X,Y, rank,sub_block, neighbors, N, sub_block_size) 
            if res:updates.append([X,Y,res[0],res[1],rank])
    return updates 

def attack_phase(rank, sub_block, neighbors, N, sub_block_size):
    res=[]
    for x in range(sub_block_size):
        for y in range(sub_block_size):
            name=sub_block[x][y].name
            if name=='.':
                continue             
            X,Y=get_global_idx(x,y,rank,sub_block_size,N)                
            tmp=attack(X,Y,rank, sub_block, neighbors, N, sub_block_size)
            if name=='A':tmp.extend(attack_second_diagonals(X,Y,rank, sub_block, neighbors, N, sub_block_size))            
            res.append([[X,Y],tmp])

    return res             

def attack(x,y,rank, sub_block, neighbors, N, sub_block_size):
    res=[]
    if check_if_can_attack:   
        name=neighbors[x][y].name                       
        for dx,dy in faction_stats[name]['attack_pattern']:
            nx,ny=x+dx,y+dy        
            if 0<=nx<N and 0<=ny<N and neighbors[nx][ny].name!='.' and neighbors[nx][ny].name!=name:
                res.append([nx,ny])
    return res

def attack_second_diagonals(x,y,rank, sub_block, neighbors, N, sub_block_size):
    res=[]    
    for dx,dy in faction_stats['A']['attack_pattern']:
        fx,fy=x+dx,y+dy 
        if 0<=fx<N and 0<=fy<N and neighbors[fx][fy].name=='.':
            sx,sy=x+2*dx,y+2*dy  
            if 0<=sx<N and 0<=sy<N and neighbors[sx][sy].name not in {'.','A'}:
                res.append([sx,sy])
    return res            

def check_if_can_attack(x,y,rank, sub_block, neighbors, N, sub_block_size):
    unit=sub_block[x][y]
    max_hp=faction_stats[unit.name]['max_health']
    cur_hp=unit.hp 
    targets=count_number_of_targets(x,y,rank,sub_block,N,sub_block_size,unit.name)
    if cur_hp<max_hp/2 or targets==0:
        return False     
    return True 

def merge_sub_blocks(sub_blocks, grid_size):
    # Initialize an empty grid
    full_grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
    
    sub_block_size = len(sub_blocks[0][0])  # Size of a single sub-block (e.g., 3x3)
    
    for sub_block, (row_block, col_block) in sub_blocks:
        # Calculate the starting position in the full grid
        start_row = row_block * sub_block_size
        start_col = col_block * sub_block_size
        
        # Insert the sub-block into the full grid
        for i in range(sub_block_size):
            for j in range(sub_block_size):
                full_grid[start_row + i][start_col + j] = sub_block[i][j]
    
    return full_grid

def get_global_idx(x, y, rank, sub_block_size, N):
    # Determine the number of sub-blocks along one side
    blocks_per_side = N // sub_block_size
    
    # Compute the sub-block row and column
    sub_block_row = (rank - 1) // blocks_per_side
    sub_block_col = (rank - 1) % blocks_per_side
    
    # Compute the global origin of the sub-block
    global_origin_x = sub_block_row * sub_block_size
    global_origin_y = sub_block_col * sub_block_size
    
    # Compute the global coordinates
    global_x = global_origin_x + x
    global_y = global_origin_y + y
    
    return [global_x, global_y]

   
def move_air_unit(x,y, rank,sub_block, neighbors, N, sub_block_size): 
    options=[]
    for nx,ny in [(x+dx, y+dy) for dx,dy in faction_stats['A']['attack_pattern']]:         
        if nx<0 or nx>=N or ny<0 or ny>=N or neighbors[nx][ny].name!='.':continue 
        targets=count_number_of_targets(nx,ny, rank, sub_block, neighbors, N, sub_block_size,'A') 
        options.append([nx,ny,targets])
    options.sort(key=lambda x:(-x[2],x[1],x[0]))
    if options and options[0][2]<=count_number_of_targets(x,y,rank,sub_block,neighbors,N,sub_block_size,'A'):
        options[0]=[x,y,0]
    return options[0][:2] if options else []   

def count_number_of_targets(x,y, rank, sub_block, neighbors, N, sub_block_size,unit):    
    tot=0   
    for nx,ny in [(x+dx, y+dy) for dx,dy in faction_stats[unit]['attack_pattern']]:
        if 0<=nx<N and 0<=ny<N and neighbors[nx][ny].name not in {'.',unit}:    
            tot+=1
    if unit!='A':
        return tot          
    for dx,dy in faction_stats[unit]['attack_pattern']:
        if 0<=x+dx<N and 0<=y+dy<N and neighbors[x+dx][y+dy].name=='.':
            if 0<=x+2*dx<N and 0<=y+2*dy<N:
                target=neighbors[x+2*dx][y+2*dy].name
                tot+=target not in {unit,'.'}
    return tot

def apply_all_updates(grid,all_updates):
    for preX,preY,newX,newY,_ in all_updates:
        grid[preX][preY],grid[newX][newY]=grid[newX][newY],grid[preX][preY]

def apply_attack(x,y,tx,ty,grid):      
    attacked=grid[tx][ty].name
    damage=grid[x][y].attack_power
    if attacked=='E':
        damage=math.floor(damage/2)        
    grid[tx][ty].hp=grid[tx][ty].hp-damage 
    return damage

def apply_attacks(attacks,grid,N):    
    record=[]
    for [[x,y],targets] in attacks:
        for tx,ty in targets:          
            damage=apply_attack(x,y,tx,ty,grid)
            record.append([x,y,tx,ty,damage])
    return record        

def check_fire_units(attacks,grid,N):
    increase_attack_power=set()
    for [(x,y),targets] in attacks:
        if grid[x][y].name!='F':
            continue
        for tx,ty in targets:
            if grid[tx][ty].hp<=0:
                increase_attack_power.add((x,y))                    
    for x,y in increase_attack_power:
        grid[x][y].attack_power=min(6,grid[x][y].attack_power+1)

def select_cell_to_flood(grid,N,x,y):
    candidates=[]
    for nx,ny in [(x,y+1),(x-1,y),(x-1,y),(x+1,y),(x-1,y-1),(x-1,y+1),(x+1,y+1),(x+1,y-1)]:
        if 0<=nx<N and 0<=ny<N and grid[nx][ny].name=='.': 
            candidates.append((nx,ny))
    candidates.sort(key=lambda x:(x[0],x[1]))
    return candidates[0] if candidates else (None,None)
        
def flood_one_cell(grid,N):
    for x in range(N):
        for y in range(N):
            unit=grid[x][y]
            if unit.name!='W':
                continue
            X,Y=select_cell_to_flood(grid,N,x,y)
            if X is not None:
                grid[X][Y]=UNIT('W',faction_stats['W']['max_health'],faction_stats['W']['attack_power'])

def remove_dead_units(grid,N):
    for x in range(N):
        for y in range(N):
            unit=grid[x][y]
            if unit.hp<=0:
                grid[x][y]=UNIT('.',0,0)   

def heal(to_heal,N,grid):
    for x,y in to_heal:
        unit=grid[x][y]
        if unit.name!='.':
            grid[x][y].hp=min(unit.hp+faction_stats[unit.name]['heal_rate'],faction_stats[unit.name]['max_health'])       

def nullify_attack_power(grid,N):
    for x in range(N):
        for y in range(N):
            unit=grid[x][y]
            if unit.name=='F':
                grid[x][y].attack_power=faction_stats['F']['attack_power']  

def visualise(grid):
    for row in grid:
        cur=[col.name for col in row]
        print(" ".join(cur))       

def get_neighbors(rank):
    pass                                                      

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processors = comm.Get_size()

input_file = "input.txt"

if rank == 0:
    N, W, T, R, waves = parse_input(input_file)  # Parse input file

    grid = [[UNIT('.', 0, 0)] * N for _ in range(N)]

    # Distribute initial grid settings to workers
    for i in range(1, num_processors):
        comm.send((N, W, T, R), dest=i, tag=0)

    for w in range(W):
        print(f"\nWAVE {w+1}:")
        wave_units = waves[w+1]  # Units for the current wave
        populate_grid(grid, N, wave_units)

        print("Grid after initialization:")
        visualise(grid)

        for round_num in range(R):
            print(f"\nROUND {round_num + 1}:")

            # Movement Phase
            all_updates = []
            blocks = split_grid(grid, num_processors)
            for rank in range(1, num_processors):
                comm.send((blocks[rank - 1], grid), dest=rank, tag=12)
                updates = comm.recv(source=rank, tag=13)
                all_updates.extend(updates)
            apply_all_updates(grid, all_updates)

            print("\tAfter windrush (Air unit movements):")
            if all_updates:
                for preX, preY, newX, newY, _ in all_updates:
                    print(f"\t\tUnit {grid[newX][newY].name} moved from ({preX}, {preY}) to ({newX}, {newY}).")
            else:
                print("\t\tNo air unit movements.")
            visualise(grid)

            # Action Phase
            new_partition = split_grid(grid, num_processors)
            attacks = []
            for r in range(1, num_processors):
                comm.send((new_partition[r - 1], grid), dest=r, tag=15)
                updates2 = comm.recv(source=r, tag=16)
                attacks.extend(updates2)

            # Apply Attacks            
            for i in range(len(attacks)):
                [x,y],_=attacks[i]
                unit=grid[x][y]
                if unit.hp<faction_stats[unit.name]['max_health']/2:
                    attacks[i][1]=[]
            record = apply_attacks(attacks, grid, N)
            check_fire_units(attacks, grid, N)

            print("\n\tAttacks:")
            if attacks:
                for [x, y, tx,ty, damage] in record:
                    attacker = grid[x][y].name
                    print(f"\t\t{attacker} unit at ({x}, {y}) attacked unit at ({tx}, {ty}) with damage: {damage}.")
            else:
                print("\t\tNo attacks this round.")

            print("\n\tHealth reductions:")
            for tx, ty in set((tx,ty) for i,j,tx,ty,_ in record):
                if grid[tx][ty].hp <= 0:
                    print(f"\t\tUnit at ({tx}, {ty}) died.")
                else:
                    print(f"\t\tUnit at ({tx}, {ty}) now has {grid[tx][ty].hp} HP.")

            # Healing Phase
            attacked=set((i,j) for i,j,_,_,_ in record)
            to_heal=set((i,j) for i in range(N) for j in range(N) if (i,j) not in attacked and grid[i][j].name!='.'
                            and grid[i][j].hp!=faction_stats[grid[i][j].name]['max_health'])            
            print("\n\tHealings:")
            for x, y in to_heal:
                print(f"\t\t{grid[x][y].name} unit at ({x}, {y}) healed to {grid[x][y].hp} HP.")

            # Cleanup Dead Units
            remove_dead_units(grid, N)
            heal(to_heal, N, grid)

            # Final Grid Visualization for the Round
            print("\n\tGrid after this round:")
            visualise(grid)

        # End-of-Wave Actions
        flood_one_cell(grid, N)
        nullify_attack_power(grid, N)
        print(f"\nGrid after wave {w + 1}:")
        visualise(grid)

    print("\nAll waves completed.")

else:      
    N, W, T, R = comm.recv(source=0, tag=0)
    sub_block_size = N // int((num_processors- 1) ** 0.5)  

    for w in range(W*R):
        sub_block,neighbors=comm.recv(source=0,tag=12)       
        updates=movement_phase(rank, sub_block[0], neighbors, N, sub_block_size)   
        comm.send(updates, dest=0, tag=13)
        sub_block2,neighbors2=comm.recv(source=0,tag=15)
        updates2=attack_phase(rank, sub_block2[0], neighbors2, N, sub_block_size)
        comm.send(updates2,dest=0,tag=16)
        
        

    

        
