# mapping of instruments to classes based on Jointist paper
import numpy as np

# key: midi index, value: class index
# midi program to our class index
program_to_index = {i: 0 for i in range(8)}                 # 0: Piano
program_to_index.update({i: 1 for i in range(8, 16)})       # 1: Chr. Percussion
program_to_index.update({i: 2 for i in range(16, 24)})      # 2: Organ
program_to_index.update({i: 3 for i in range(24, 32)})      # 3: Guitar
program_to_index.update({i: 4 for i in range(32, 40)})      # 4: Bass
program_to_index.update({i: 5 for i in range(40, 48)})      # 5: Strings (exclude timpani)
program_to_index.update({i: 5 for i in range(49, 52)})      # 5: Strings (ensemble)
program_to_index.update({i: 6 for i in range(52, 55)})      # 6: Voice (exclude orchestral hit from ensemble)
program_to_index.update({i: 7 for i in range(56, 64)})      # 7: Brass
program_to_index.update({i: 8 for i in range(64, 72)})      # 8: Reed
program_to_index.update({i: 9 for i in range(72, 80)})      # 9: Pipe
program_to_index.update({i: 10 for i in range(80, 88)})     # 10: Synth Lead
program_to_index.update({i: 11 for i in range(88, 96)})     # 11: Synth Pad
program_to_index.update({i: 12 for i in range(104, 112)})   # 12: Ethnic

# midi program to our class name
# key: midi index, value: our class name
program_to_name = {i: 'Piano' for i in range(8)}                                # 0: Piano
program_to_name.update({i: 'Chr Percussion' for i in range(8, 16)})            # 1: Chr Percussion
program_to_name.update({i: 'Organ' for i in range(16, 24)})                     # 2: Organ
program_to_name.update({i: 'Guitar' for i in range(24, 32)})                    # 3: Clavinet
program_to_name.update({i: 'Bass' for i in range(32, 40)})                      # 4: Bass
program_to_name.update({i: 'Strings' for i in range(40, 52)})                   # 5: Strings
program_to_name.update({i: 'Voice' for i in range(52, 55)})                     # 6: Voice
program_to_name.update({i: 'Brass' for i in range(56, 64)})                     # 7: Brass
program_to_name.update({i: 'Reed' for i in range(64, 72)})                      # 8: Reed
program_to_name.update({i: 'Pipe' for i in range(72, 80)})                      # 9: Pipe
program_to_name.update({i: 'Synth Lead' for i in range(80, 88)})                # 10: Synth Lead
program_to_name.update({i: 'Synth Pad' for i in range(88, 96)})                 # 11: Synth Pad
program_to_name.update({i: 'Ethnic' for i in range(104, 112)})                  # 12: Ethnic

# index to midi program name
index_to_name = {}
for k, v in program_to_name.items():
    program = k
    if program in program_to_index.keys():
        index = program_to_index[program]
        index_to_name[index] = v

name_to_range = dict(
Bass=(1, 115),
Brass=(0, 109),
Ethnic=(36, 116),
Guitar=(1, 126),
Organ=(10, 114),
Piano=(0, 127),
Pipe=(12, 127),
Reed=(0, 123),
Strings=(18, 125),
Voice=(12, 100)
)
name_to_range['Synth Lead'] = (8, 120)
name_to_range['Synth Pad'] = (21, 118)
name_to_range['Chr Percussion'] = (28, 121)

# instruments
instruments = np.unique(list(program_to_name.values()))

# todo: maybe replace brass, reed, pipe with winds
