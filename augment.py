import random
import os
import re
import numpy as np
import torch
import scipy.io as scio

# receiver locations
map_positions = {
    1: (0.5, -0.5),  
    2: (1.4, -0.5), 
    3: (2.0, 0.0), 
    4: (-0.5, 0.5),
    5: (-0.5, 1.4),
    6: (0.0, 2.0),
}
tx_position = (0., 0.)  # transmitter location

virtual_positions = {
    1: (0.5, 0.0),
    2: (1.4, 0.0),
    3: (1.0, 0.5),
    4: (1.0, 1.5),
    5: (2.0, 1.0),
    6: (2.0, 2.0),
}
d_max = 2.8284  # maximum distance rx = (2.0, 2.0)


def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# generate weighted virtual CSI from ref_data
def augment_csi(ref_data, vir_id, gamma=2):
    feature = ref_data['feature']
    cond = ref_data['cond'].squeeze(0)

    receiver_id = int(cond[-2])
    rx_position = map_positions[receiver_id]
    d = calculate_distance(rx_position, tx_position)
    
    # generate virtual CSI
    virtual_rx_position = virtual_positions[vir_id]
    d_prime = calculate_distance(virtual_rx_position, tx_position)
    scale = (d / d_prime) ** gamma
    
    new_feature = feature * scale  # rescale amplitude
    
    new_cond = cond
    new_cond[-2] = vir_id + 6
    
    weight = np.abs(d - d_prime)
    return new_feature, [new_cond], weight


# cond_data = [room_id, gesture_type, torso_location, face_orientation, receiver_id, user_id]
# file_name = user_id-gesture_type-torso_location-face_orientation-repetition_number-receiver_id.mat

# generate virtual CSI from all receiver locations
def virtual_augmentation(dir, cond, gamma=2):
    user_id , gesture_type, torso_location, face_orientation, repeat = cond
    cond_str = f'user{user_id}-{gesture_type}-{torso_location}-{face_orientation}-{repeat}'

    ref_files = []
    for file in os.listdir(dir):
        if cond_str in file and file.endswith('.mat'):
            ref_files.append(file)

    for vir_id in range(1, 7):
        features = []
        weights = []
        
        for ref_file in ref_files:
            ref_data = scio.loadmat(os.path.join(dir, ref_file), verify_compressed_data_integrity=False)
            
            data, cond, weight = augment_csi(ref_data, vir_id, gamma)
            features.append(data)
            weights.append(weight)
        
        # weighted average of virtual CSI
        min_length = min(feature.shape[0] for feature in features)
        truncated_features = [feature[:min_length] for feature in features]
        features = np.array(truncated_features)
        weights = 1 / (1 + np.array(weights))
        weighted_avg_feature = np.average(features, axis=0, weights=weights)
        
        # save virtual CSI
        file_name = f'user{user_id}-{gesture_type}-{torso_location}-{face_orientation}-{repeat}-r{vir_id+6}.mat'
        save_dir = '/data/Widar3.0_aug' + dir.split('/data/Widar3.0')[-1]
        file_path = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        scio.savemat(file_path, {'feature': weighted_avg_feature, 'cond': cond})


def extract_abcd(filename):
    pattern = r'user\d+-(\d+)-(\d+)-(\d+)-(\d+)-r\d+\.mat'
    match = re.match(pattern, filename)
    if match:
        a, b, c, d = map(int, match.groups())
        return a, b, c, d
    return None

def find_max_abcd_file(directory):
    max_abcd = None
    max_file = None
    
    for filename in os.listdir(directory):
        abcd = extract_abcd(filename)
        if abcd:
            if max_abcd is None or abcd > max_abcd:
                max_abcd = abcd
                max_file = filename
    
    return max_file

# generate virtual CSI of all conditions
def generate_virtual_csi(gamma=2):
    dir = '/data/Widar3.0/20181208/user3'
    max_file = find_max_abcd_file(dir)

    file_parts = max_file.split('-')
    user_id = int(file_parts[0].replace('user', ''))  # user id
    gesture_idx = int(file_parts[1])
    torso_location = int(file_parts[2])
    face_orientation = int(file_parts[3])
    repetition_number = int(file_parts[4])

    for ges in range(1, gesture_idx + 1):
        for torso in range(1, torso_location + 1):
            for face in range(1, face_orientation + 1):
                for repeat in range(1, repetition_number + 1):
                    print(f'Generating virtual CSI for user{user_id}-{ges}-{torso}-{face}-{repeat}')
                    cond = [user_id, ges, torso, face, repeat]
                    virtual_augmentation(dir, cond, gamma)

if __name__ == '__main__':
    generate_virtual_csi(gamma=2)