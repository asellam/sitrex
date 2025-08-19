import json, os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Function to load JSON data from a file
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def point(landmark):
    return np.array([landmark['x'], landmark['y'], landmark['z']])

# Function to calculate the 3D angle between three points
def angle_between(a, b):
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def vector(start, end):
    return np.array([end['x'] - start['x'], end['y'] - start['y'], end['z'] - start['z']])

def calculate_angle(a, b, c):
    return angle_between(vector(b, a), vector(b, c))

def midpoint(a, b):
    return (a + b) / 2

def vector_plane_project(vector, plane_normal):
    vector_proj = vector - np.dot(vector, plane_normal) * plane_normal
    vector_proj = normalize(vector_proj)
    return vector_proj


# Function to compute 9 specific angles from 33 keypoints
def calculate_angles(landmarks: list) -> list:
    """Compute 20 joint-angle features from Mediapipe landmarks."""
    if not landmarks or len(landmarks) < 33:
        return [0.0] * NUM_FEATURES
    # Define landmark indices
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_PINKY, RIGHT_PINKY = 17, 18
    LEFT_INDEX,RIGHT_INDEX = 19, 20
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28

    # Estimating Primary Vectors
    L_shoulder = point(landmarks[LEFT_SHOULDER])  # LEFT_SHOULDER
    R_shoulder = point(landmarks[RIGHT_SHOULDER])  # RIGHT_SHOULDER
    L_hip = point(landmarks[LEFT_HIP])       # LEFT_HIP
    R_hip = point(landmarks[RIGHT_HIP])       # RIGHT_HIP
    L_ankle = point(landmarks[LEFT_ANKLE])       # LEFT_ANKLE
    R_ankle = point(landmarks[RIGHT_ANKLE])       # RIGHT_ANKLE

    # Midpoints
    mid_shoulder = midpoint(L_shoulder, R_shoulder)
    mid_hip = midpoint(L_hip, R_hip)

    # Local torso axes
    horizontal = normalize(R_shoulder - L_shoulder)           # Local X (shoulder width)
    upward = normalize(mid_shoulder - mid_hip)          # Local Y (torso up)
    forward = normalize(np.cross(horizontal, upward))          # Local Z (front-back normal to torso plane)
    # Re-orthogonalize Y just in case (to maintain right-handed frame)
    upward = normalize(np.cross(forward, horizontal))

    # Torso flexion
    mid_ankle = midpoint(L_ankle, R_ankle)
    torso = mid_shoulder - mid_hip
    gravity = mid_ankle - mid_hip
    torso_flexion = angle_between(torso, gravity)

    # Elbow
    left_elbow = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
    right_elbow = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])

    # Shoulder
    left_shoulder = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW])
    right_shoulder = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW])

    # Spinal alignment
    left_hand = midpoint(point(landmarks[LEFT_PINKY]), point(landmarks[LEFT_INDEX]))
    left_wrist = angle_between(left_hand - point(landmarks[LEFT_WRIST]),
                               point(landmarks[LEFT_ELBOW]) - point(landmarks[LEFT_WRIST]))
    right_hand = midpoint(point(landmarks[RIGHT_PINKY]), point(landmarks[RIGHT_INDEX]))
    right_wrist = angle_between(right_hand - point(landmarks[RIGHT_WRIST]),
                                point(landmarks[RIGHT_ELBOW]) - point(landmarks[RIGHT_WRIST]))

    # Scapular upward rotation
    left_scapular_upward_rotation = angle_between(vector(landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER]),
                                                  vector(landmarks[LEFT_HIP], landmarks[RIGHT_HIP]))

    right_scapular_upward_rotation = angle_between(vector(landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER]),
                                                   vector(landmarks[RIGHT_HIP], landmarks[LEFT_HIP]))

    # Shoulder abduction/extension
    v_arm_left = vector(landmarks[LEFT_SHOULDER],  landmarks[LEFT_ELBOW])
    f_proj_v_arm_left = vector_plane_project(v_arm_left, forward)
    f_left_shoulder_abduction = angle_between(f_proj_v_arm_left, upward)
    h_proj_v_arm_left = vector_plane_project(v_arm_left, upward)
    h_left_shoulder_adduction = angle_between(h_proj_v_arm_left, forward)
    s_proj_v_arm_left = vector_plane_project(v_arm_left, horizontal)
    left_shoulder_extension = angle_between(s_proj_v_arm_left, upward)

    v_arm_right = vector(landmarks[RIGHT_SHOULDER],  landmarks[RIGHT_ELBOW])
    f_proj_v_arm_right = vector_plane_project(v_arm_right, forward)
    f_right_shoulder_abduction = angle_between(f_proj_v_arm_right, upward)
    h_proj_v_arm_right = vector_plane_project(v_arm_right, upward)
    h_right_shoulder_adduction = angle_between(h_proj_v_arm_right, forward)
    s_proj_v_arm_right = vector_plane_project(v_arm_right, horizontal)
    right_shoulder_extension = angle_between(s_proj_v_arm_right, upward)

    # Knee
    left_knee = calculate_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
    right_knee = calculate_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])

    # Hip
    left_hip = calculate_angle(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], landmarks[LEFT_KNEE])
    right_hip = calculate_angle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE])

    # Ankle dorsiflexion
    v_shin_left = vector(landmarks[LEFT_KNEE],  landmarks[LEFT_ANKLE])
    s_proj_v_shin_left = vector_plane_project(v_shin_left, horizontal)
    left_ankle_dorsiflexion = angle_between(s_proj_v_shin_left, upward)

    v_shin_right = vector(landmarks[RIGHT_KNEE],  landmarks[RIGHT_ANKLE])
    s_proj_v_shin_right = vector_plane_project(v_shin_right, horizontal)
    right_ankle_dorsiflexion = angle_between(s_proj_v_shin_right, upward)

    # Left knee valgus indicator
    f_proj_left_hip = vector_plane_project(point(landmarks[LEFT_HIP]), forward)
    f_proj_left_knee = vector_plane_project(point(landmarks[LEFT_KNEE]), forward)
    f_proj_left_ankle = vector_plane_project(point(landmarks[LEFT_ANKLE]), forward)
    left_knee_valgus_indicator = angle_between(f_proj_left_hip - f_proj_left_knee, f_proj_left_ankle - f_proj_left_knee)
    # Right knee valgus indicator
    f_proj_right_hip = vector_plane_project(point(landmarks[RIGHT_HIP]), forward)
    f_proj_right_knee = vector_plane_project(point(landmarks[RIGHT_KNEE]), forward)
    f_proj_right_ankle = vector_plane_project(point(landmarks[RIGHT_ANKLE]), forward)
    right_knee_valgus_indicator = angle_between(f_proj_right_hip - f_proj_right_knee, f_proj_right_ankle - f_proj_right_knee)

    angles = {
        'torso flexion': torso_flexion,
        'left elbow': left_elbow,
        'right elbow': right_elbow,
        'left shoulder': left_shoulder,
        'right shoulder': right_shoulder,
        'left wrist': left_wrist,
        'right wrist': right_wrist,
        'left frontal shoulder abduction': f_left_shoulder_abduction,
        'right frontal shoulder abduction': f_right_shoulder_abduction,
        'left scapular upward rotation': left_scapular_upward_rotation,
        'right scapular upward rotation': right_scapular_upward_rotation,
        'left knee': left_knee,
        'right knee': right_knee,
        'left hip': left_hip,
        'right hip': right_hip,
        'left ankle dorsiflexion': left_ankle_dorsiflexion,
        'right ankle dorsiflexion': right_ankle_dorsiflexion,
        'left knee valgus indicator': left_knee_valgus_indicator,
        'right knee valgus indicator': right_knee_valgus_indicator,
        'left horizontal shoulder adduction': h_left_shoulder_adduction,
        'right horizontal shoulder adduction': h_right_shoulder_adduction,
        'left shoulder extension': left_shoulder_extension,
        'right shoulder extension': right_shoulder_extension,
    }

    return angles
import json

# Function to extract only angles from video frames
def extract_landmarks(data, all_angles):
    frames = data['frames']
    angles_per_frame = []
    for frame in frames:
        landmarks = frame['landmarks']
        if landmarks and len(landmarks) >= 33:
            angles_dict = calculate_angles(landmarks)
            # Add angles with a fixed order (each angle will have the same order in all sequences)
            angles = [angles_dict[angle] for angle in all_angles]
            angles_per_frame.append(angles)
    return np.array(angles_per_frame)

# Function to load the dataset from the directory
def load_dataset(base_dir):
    all_sequences = []
    labels = []
    json_files = [
        (d, f)
        for d in os.listdir(base_dir) if d.endswith('_json')
        for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.json')
    ]
    exercise_angles = load_json_data(os.path.join(base_dir, 'exercise_angles.json'))
    # Get a list of all angle names without duplicates
    all_angles = sorted(list(set([angle for exercise in exercise_angles for angle in exercise_angles[exercise]])))
    for exercise_dir, json_file in tqdm(json_files, desc="Loading dataset"):
        exercise_label = exercise_dir.replace('_json', '').replace('_', ' ')
        json_path = os.path.join(base_dir, exercise_dir, json_file)
        data = load_json_data(json_path)
        sequence = extract_landmarks(data, all_angles)
        if sequence.size > 0:
            all_sequences.append(sequence)
            labels.append(exercise_label)
    print(f"Loaded {len(all_sequences)} sequences with labels.")
    # Replace angle names by their index in the sequence array last axis
    for exercise in exercise_angles:
        for i in range(len(exercise_angles[exercise])):
            angle = exercise_angles[exercise][i]
            exercise_angles[exercise][i] = all_angles.index(angle)
    return all_sequences, labels, exercise_angles

# Function to augment a sequence by adding noise
def augment_and_pad(sequence, maxlen, train):
    length = min(np.random.randint(maxlen//2, maxlen), sequence.shape[0])
    offset = np.random.randint(0, sequence.shape[0] - length + 1)
    sequence = sequence[offset:offset+length]
    if train:
        noise = np.random.normal(0, 5, length)
        sequence = sequence + noise
    padded = np.zeros(maxlen, sequence.dtype)
    padded[:length] = sequence[:]
    return padded

# Function to apply temporal smoothing to sequences
def temporal_smoothing(sequences, window_size=5):
    smoothed = []
    for seq in sequences:
        smoothed_seq = []
        for i in range(len(seq)):
            window = seq[max(0, i - window_size // 2): i + window_size // 2 + 1]
            avg_frame = np.mean(window, axis=0)
            smoothed_seq.append(avg_frame)
        smoothed.append(smoothed_seq)
    return smoothed

# Function to preprocess sequences: smooth, resample, augment, and pad
def preprocess_data(sequences, labels, exercise_angles):
    exercises = sorted(list(set(labels)))
    smoothed_sequences = temporal_smoothing(sequences)
    all_sequences =  [np.array(seq) for seq in smoothed_sequences]
    label_to_index = {label: idx for idx, label in enumerate(exercises)}
    numerical_labels = [label_to_index[label] for label in labels]
    label_angles = [exercise_angles[exercise] for exercise in exercises]
    return all_sequences, numerical_labels, label_angles

def normalize_angle_points(points):
    a, b, c = points
    la = np.linalg.norm(a - b)
    lc = np.linalg.norm(c - b)
    lm = max(la, lc)
    d = b + (a - b) * lm / la * 0.1 
    e = b + (c - b) * lm / lc * 0.1
    a = b + (a - b) * lm / la * 0.3 
    c = b + (c - b) * lm / lc * 0.3
    return b, a, c, d, e

# Function to compute 23 specific angles from 33 keypoints
def get_angle_points(landmarks):
    """Compute 20 joint-angle features from Mediapipe landmarks."""
    if not landmarks or len(landmarks) < 33:
        return [0.0] * NUM_FEATURES
    # Define landmark indices
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_PINKY, RIGHT_PINKY = 17, 18
    LEFT_INDEX,RIGHT_INDEX = 19, 20
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28

    # Estimating Primary Vectors
    L_shoulder = point(landmarks[LEFT_SHOULDER])  # LEFT_SHOULDER
    R_shoulder = point(landmarks[RIGHT_SHOULDER])  # RIGHT_SHOULDER
    L_hip = point(landmarks[LEFT_HIP])       # LEFT_HIP
    R_hip = point(landmarks[RIGHT_HIP])       # RIGHT_HIP
    L_ankle = point(landmarks[LEFT_ANKLE])       # LEFT_ANKLE
    R_ankle = point(landmarks[RIGHT_ANKLE])       # RIGHT_ANKLE

    # Midpoints
    mid_shoulder = midpoint(L_shoulder, R_shoulder)
    mid_hip = midpoint(L_hip, R_hip)

    # Local torso axes
    horizontal = normalize(R_shoulder - L_shoulder)           # Local X (shoulder width)
    upward = normalize(mid_shoulder - mid_hip)          # Local Y (torso up)
    forward = normalize(np.cross(horizontal, upward))          # Local Z (front-back normal to torso plane)
    # Re-orthogonalize Y just in case (to maintain right-handed frame)
    upward = normalize(np.cross(forward, horizontal))

    # Torso flexion
    mid_ankle = midpoint(L_ankle, R_ankle)
    torso = mid_shoulder - mid_hip
    gravity = mid_ankle - mid_hip
    torso_flexion = (mid_shoulder, mid_hip, mid_ankle)
    
    # Elbow
    left_elbow = (point(landmarks[LEFT_WRIST]), point(landmarks[LEFT_ELBOW]), point(landmarks[LEFT_SHOULDER]))
    right_elbow = (point(landmarks[RIGHT_WRIST]), point(landmarks[RIGHT_ELBOW]), point(landmarks[RIGHT_SHOULDER]))

    # Shoulder
    left_shoulder = (point(landmarks[LEFT_HIP]), point(landmarks[LEFT_SHOULDER]), point(landmarks[LEFT_ELBOW]))
    right_shoulder = (point(landmarks[RIGHT_HIP]), point(landmarks[RIGHT_SHOULDER]), point(landmarks[RIGHT_ELBOW]))

    # Wrist
    left_hand = midpoint(point(landmarks[LEFT_PINKY]), point(landmarks[LEFT_INDEX]))
    left_wrist = (left_hand, point(landmarks[LEFT_WRIST]), point(landmarks[LEFT_ELBOW]))
    right_hand = midpoint(point(landmarks[RIGHT_PINKY]), point(landmarks[RIGHT_INDEX]))
    right_wrist = (right_hand, point(landmarks[RIGHT_WRIST]), point(landmarks[RIGHT_ELBOW]))

    # Scapular upward rotation
    left_scapular_upward_rotation = (point(landmarks[LEFT_SHOULDER]), point(landmarks[LEFT_HIP]),  point(landmarks[RIGHT_HIP]))
    
    right_scapular_upward_rotation = (point(landmarks[RIGHT_SHOULDER]), point(landmarks[RIGHT_HIP]),  point(landmarks[LEFT_HIP]))

    # Shoulder abduction/extension
    shoulder = point(landmarks[LEFT_SHOULDER])
    v_arm_left = vector(landmarks[LEFT_SHOULDER],  landmarks[LEFT_ELBOW])
    f_proj_v_arm_left = vector_plane_project(v_arm_left, forward)
    f_left_shoulder_abduction = (shoulder + f_proj_v_arm_left, shoulder, shoulder + upward)
    h_proj_v_arm_left = vector_plane_project(v_arm_left, upward)
    h_left_shoulder_adduction = (shoulder + h_proj_v_arm_left, shoulder, shoulder + forward)
    s_proj_v_arm_left = vector_plane_project(v_arm_left, horizontal)
    left_shoulder_extension = (shoulder + s_proj_v_arm_left, shoulder, shoulder + upward)
    
    shoulder = point(landmarks[RIGHT_SHOULDER])
    v_arm_right = vector(landmarks[RIGHT_SHOULDER],  landmarks[RIGHT_ELBOW])
    f_proj_v_arm_right = vector_plane_project(v_arm_right, forward)
    f_right_shoulder_abduction = (shoulder + f_proj_v_arm_right, shoulder, shoulder + upward)
    h_proj_v_arm_right = vector_plane_project(v_arm_right, upward)
    h_right_shoulder_adduction = (shoulder + h_proj_v_arm_right, shoulder, shoulder + forward)
    s_proj_v_arm_right = vector_plane_project(v_arm_right, horizontal)
    right_shoulder_extension = (shoulder + s_proj_v_arm_right, shoulder, shoulder + upward)

    # Knee
    left_knee = (point(landmarks[LEFT_HIP]), point(landmarks[LEFT_KNEE]), point(landmarks[LEFT_ANKLE]))
    right_knee = (point(landmarks[RIGHT_HIP]), point(landmarks[RIGHT_KNEE]), point(landmarks[RIGHT_ANKLE]))

    # Hip
    left_hip = (point(landmarks[LEFT_SHOULDER]), point(landmarks[LEFT_HIP]), point(landmarks[LEFT_KNEE]))
    right_hip = (point(landmarks[RIGHT_SHOULDER]), point(landmarks[RIGHT_HIP]), point(landmarks[RIGHT_KNEE]))
    
    # Ankle dorsiflexion
    ankle = point(landmarks[LEFT_ANKLE])
    v_shin_left = vector(landmarks[LEFT_KNEE],  landmarks[LEFT_ANKLE])
    s_proj_v_shin_left = vector_plane_project(v_shin_left, horizontal)
    left_ankle_dorsiflexion = (ankle + s_proj_v_shin_left, ankle, ankle + upward)

    ankle = point(landmarks[RIGHT_ANKLE])
    v_shin_right = vector(landmarks[RIGHT_KNEE],  landmarks[RIGHT_ANKLE])
    s_proj_v_shin_right = vector_plane_project(v_shin_right, horizontal)
    right_ankle_dorsiflexion = (ankle + s_proj_v_shin_right, ankle, ankle + upward)
    
    # Left knee valgus indicator
    knee = point(landmarks[LEFT_KNEE])
    f_proj_left_hip = vector_plane_project(point(landmarks[LEFT_HIP]), forward)
    f_proj_left_knee = vector_plane_project(point(landmarks[LEFT_KNEE]), forward)
    f_proj_left_ankle = vector_plane_project(point(landmarks[LEFT_ANKLE]), forward)
    left_knee_valgus_indicator = (knee + f_proj_left_hip - f_proj_left_knee, knee, knee + f_proj_left_ankle - f_proj_left_knee)
    # Right knee valgus indicator
    knee = point(landmarks[RIGHT_KNEE])
    f_proj_right_hip = vector_plane_project(point(landmarks[RIGHT_HIP]), forward)
    f_proj_right_knee = vector_plane_project(point(landmarks[RIGHT_KNEE]), forward)
    f_proj_right_ankle = vector_plane_project(point(landmarks[RIGHT_ANKLE]), forward)
    right_knee_valgus_indicator = (knee + f_proj_right_hip - f_proj_right_knee, knee, knee + f_proj_right_ankle - f_proj_right_knee)

    angles = {
        'torso flexion': torso_flexion,
        'left elbow': left_elbow,
        'right elbow': right_elbow,
        'left shoulder': left_shoulder,
        'right shoulder': right_shoulder,
        'left wrist': left_wrist,
        'right wrist': right_wrist,
        'left frontal shoulder abduction': f_left_shoulder_abduction,
        'right frontal shoulder abduction': f_right_shoulder_abduction,
        'left scapular upward rotation': left_scapular_upward_rotation,
        'right scapular upward rotation': right_scapular_upward_rotation,
        'left knee': left_knee,
        'right knee': right_knee,
        'left hip': left_hip,
        'right hip': right_hip,
        'left ankle dorsiflexion': left_ankle_dorsiflexion,
        'right ankle dorsiflexion': right_ankle_dorsiflexion,
        'left knee valgus indicator': left_knee_valgus_indicator,
        'right knee valgus indicator': right_knee_valgus_indicator,
        'left horizontal shoulder adduction': h_left_shoulder_adduction,
        'right horizontal shoulder adduction': h_right_shoulder_adduction,
        'left shoulder extension': left_shoulder_extension,
        'right shoulder extension': right_shoulder_extension,
    }
    for key in angles:
        angles[key] = normalize_angle_points(angles[key])
    return angles

# Custom data generator for usefulness model
class UsefulnessDataset(tf.keras.utils.Sequence):
    def __init__(self, data_x, data_y, exercises, batch_size, maxlen, train, **kwargs):
        super().__init__(**kwargs)
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.train = train
        self.maxlen = maxlen
        self.batch_x = np.empty((self.batch_size, self.maxlen, data_y.shape[1]), dtype=np.float32)
        self.batch_y = np.empty((self.batch_size, data_y.shape[1]), dtype=np.float32)
        if self.train:
            self.exercise_samples = defaultdict(lambda : [])
            for i in range(exercises.shape[0]):
                self.exercise_samples[exercises[i]].append(i)
            self.max_samples = max(len(self.exercise_samples[exercise]) for exercise in  self.exercise_samples)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.train:
            # Exercise Data Balancing (EDB)
            self.indexes = np.empty(self.max_samples*len(self.exercise_samples), np.int32)
            i = 0
            for exercise in self.exercise_samples:
                for j in self.exercise_samples[exercise]:
                    self.indexes[i] = j
                    i += 1
                for _ in range(self.max_samples-len(self.exercise_samples[exercise])):
                    j = np.random.randint(0, len(self.exercise_samples[exercise]))
                    self.indexes[i] = self.exercise_samples[exercise][j]
                    i += 1
            np.random.shuffle(self.indexes)
        else:
            self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return self.indexes.shape[0] // self.batch_size


    def __getitem__(self, index):
        return self.__data_generation(index)

    def __data_generation(self, index):
        for i in range(self.batch_size):
            j = self.indexes[index * self.batch_size + i]
            for angle in range(self.data_y.shape[1]):
                self.batch_x[i, :, angle] = augment_and_pad(self.data_x[j][:, angle], self.maxlen, self.train)
                self.batch_y[i, angle] = self.data_y[j, angle]
        return self.batch_x, self.batch_y

# Custom data generator for Similarity model
class SimilarityDataset(tf.keras.utils.Sequence):
    def __init__(self, data_x, data_y, num_batches, batch_size, label_angles, maxlen, train, **kwargs):
        super().__init__(**kwargs)
        self.data = defaultdict(lambda:[])
        for x, y in zip(data_x, data_y):
            self.data[y].append(x)
        self.num_angles = len(data_x[0][0])
        self.label_angles = label_angles
        # Stores for each angles the list of its related exercise (used to generate negative labels)
        self.maxlen = maxlen
        self.labels = list([label for label in self.data.keys() if len(self.data[label]) > 0])
        self.nb_batches = num_batches
        self.batch_size = batch_size
        self.train = train
        self.batch_left = np.empty((self.batch_size, self.maxlen, self.num_angles), dtype=np.float32)
        self.batch_right = np.empty((self.batch_size, self.maxlen, self.num_angles), dtype=np.float32)
        self.batch_target = np.empty((self.batch_size, self.num_angles), dtype=np.float32)


    def __len__(self):
        if self.nb_batches > 0:
            return self.nb_batches
        return len(self.all_pairs) // self.batch_size


    def __getitem__(self, index):
        return self.__data_generation(index)

    def __data_generation(self, index):
        for i in range(self.batch_size):
            left_label = self.labels[np.random.randint(0, len(self.labels))]
            left_idx = np.random.randint(0, len(self.data[left_label]))
            for angle in range(self.num_angles):
                if angle in self.label_angles[left_label]:
                    # positive pair
                    right_idx = np.random.randint(0, len(self.data[left_label]))
                    self.batch_left[i, :, angle] = augment_and_pad(self.data[left_label][left_idx][:, angle], self.maxlen, self.train)
                    self.batch_right[i, :, angle] = augment_and_pad(self.data[left_label][right_idx][:, angle], self.maxlen, self.train)
                    self.batch_target[i, angle] = 1
                else:
                    # negative pair
                    left_label_idx = self.labels.index(left_label)
                    right_label_idx = (left_label_idx + np.random.randint(1, len(self.labels))) % len(self.labels)
                    right_label = self.labels[right_label_idx]
                    right_idx = np.random.randint(0, len(self.data[right_label]))
                    self.batch_left[i, :, angle] = augment_and_pad(self.data[left_label][left_idx][:, angle], self.maxlen, self.train)
                    self.batch_right[i, :, angle] = augment_and_pad(self.data[right_label][right_idx][:, angle], self.maxlen, self.train)
                    self.batch_target[i, angle] = 0
        return (self.batch_left, self.batch_right), self.batch_target
