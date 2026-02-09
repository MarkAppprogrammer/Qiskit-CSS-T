import os
import sys
import numpy as np
import pandas as pd
import galois
import sinter
import glob
from typing import List, Tuple

# --- PATH SETUP FOR DEPENDENCIES ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DOUBLING_CSST_PATH = os.path.join(PROJECT_ROOT, "doubling-CSST")

if os.path.exists(DOUBLING_CSST_PATH):
    if DOUBLING_CSST_PATH not in sys.path:
        sys.path.append(DOUBLING_CSST_PATH)

# --- IMPORTS ---
try:
    from convert_alist import readAlist
except ImportError:
    def readAlist(path):
        raise ImportError(f"Could not import 'convert_alist.py'. Checked path: {DOUBLING_CSST_PATH}")

try:
    from qcodeplot3d.cc_3d import cubic_3d_dual_graph, tetrahedron_3d_dual_graph
    from qcodeplot3d.cc_2d import triangular_2d_dual_graph, square_2d_dual_graph
    from qcodeplot3d.common.stabilizers import get_check_matrix
    HAS_QCODEPLOT = True
except ImportError:
    HAS_QCODEPLOT = False

try:
    from mwpf.sinter_decoders import SinterMWPFDecoder 
except ImportError:
    pass

# --- SHARED CONFIGURATION ---
BASE_ALIST_DIR_DEV = "../../../doubling-CSST/alistMats/"
BASE_ALIST_DIR_LOCAL = "./alistMats/"
BASE_ALIST_DIR_PROJECT = os.path.join(DOUBLING_CSST_PATH, "alistMats/")
BASE_ALIST_PATH = BASE_ALIST_DIR_LOCAL if os.path.exists(BASE_ALIST_DIR_LOCAL) else (BASE_ALIST_DIR_PROJECT if os.path.exists(BASE_ALIST_DIR_PROJECT) else BASE_ALIST_DIR_DEV)

# 1. Existing Codes
OP1_DIR = os.path.join(BASE_ALIST_PATH, "GO03_self_dual/")
OP1_DICT = {4:2, 6:2, 8:2, 10:2, 12:4, 14:4, 16:4, 18:4, 20:4, 22:6, 24:6, 26:6, 28:6, 30:6, 32:8, 34:6, 36:8, 38:8, 40:8, 42:8, 44:8, 46:8, 48:8, 50:8, 52:10, 54:8, 56:10, 58:10, 60:12, 62:10, 64:10}

OP2_DIR = os.path.join(BASE_ALIST_PATH, "QR_dual_containing/")
OP2_DICT = {7:3, 17:5, 23:7, 47:11, 79:15, 103:19, 167:23}

# 2. NEW: JA25 Triorthogonal Codes
OP3_DIR = os.path.join(BASE_ALIST_PATH, "JA25_triorthogonal/")
OP3_DICT = {15: 3, 49: 5, 95: 7, 185: 9, 189: 9, 279: 11}

CODE_CONFIGS = {
    "self_dual": { 
        "source": "local", "dir": OP1_DIR, "dist_dict": OP1_DICT, "if_self_dual": True, 
        "iter_list": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64], 
        "alist_suffix": ".alist" 
    },
    "dual_containing": { 
        "source": "local", "dir": OP2_DIR, "dist_dict": OP2_DICT, "if_self_dual": True, 
        "iter_list": [7, 23, 47, 79, 103, 167], 
        "alist_suffix": ".alist" 
    },
    "triorthogonal": {
        "source": "local", 
        "dir": OP3_DIR, 
        "dist_dict": OP3_DICT, 
        "if_self_dual": False,
        "iter_list": [15, 49, 95, 185, 189, 279],
        "alist_suffix": ".alist"
    },
    "cubic": { "source": "qcodeplot3d", "type": "cubic", "iter_list": [3, 5, 7, 9] },
    "tetrahedral": { "source": "qcodeplot3d", "type": "tetrahedral", "iter_list": [3, 5, 7] },
    "triangular": { "source": "qcodeplot3d", "type": "triangular", "iter_list": [3, 5, 7, 9, 11] },
    "square": { "source": "qcodeplot3d", "type": "square", "iter_list": [3, 5, 7, 9, 11] }
}

def _extract_geometric_matrices(code_type: str, distance: int) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_QCODEPLOT:
        raise ImportError(f"Cannot generate {code_type}: qcodeplot3d package is not installed.")

    if code_type == "cubic": graph = cubic_3d_dual_graph(distance)
    elif code_type == "tetrahedral": graph = tetrahedron_3d_dual_graph(distance)
    elif code_type == "triangular": graph = triangular_2d_dual_graph(distance)
    elif code_type == "square": graph = square_2d_dual_graph(distance)
    else: raise ValueError(f"Unknown code type: {code_type}")

    x_stabilizers = [node.stabilizer for node in graph.nodes() if node.is_stabilizer]
    if code_type in ["cubic", "tetrahedral"]:
        z_stabilizers = [edge.stabilizer for edge in graph.edges() if edge.is_stabilizer]
        Hx = get_check_matrix(x_stabilizers, only_x=True)
        Hz = get_check_matrix(z_stabilizers, only_z=True)
    else:
        Hx = get_check_matrix(x_stabilizers, only_x=True)
        Hz = Hx

    return np.array(Hx, dtype=np.uint8), np.array(Hz, dtype=np.uint8)

def get_parity_matrices(val: int, config: dict):
    if config.get('source') == 'qcodeplot3d':
        return _extract_geometric_matrices(config['type'], val)

    n = val
    d = config['dist_dict'].get(n, 0)
    base_dir = config['dir']
    if_self_dual = config['if_self_dual']
    suffix = config.get('alist_suffix', '.alist')
    F2 = galois.GF(2)
    
    if if_self_dual:
        filename = f"n{n}_d{d}{suffix}"
        alistFilePath = os.path.join(base_dir, filename)
        if not os.path.exists(alistFilePath): 
            raise FileNotFoundError(f"Missing self-dual file: {alistFilePath}")
        GenMat = F2(readAlist(alistFilePath))
        G_punctured = GenMat[:, :-1]
        Hz = Hx = G_punctured.null_space() 
    else:
        path_x = os.path.join(base_dir, f"n{n}_d{d}_Hx{suffix}")
        path_z = os.path.join(base_dir, f"n{n}_d{d}_Hz{suffix}")
        
        if os.path.exists(path_x) and os.path.exists(path_z):
            Hx = F2(readAlist(path_x))
            Hz = F2(readAlist(path_z))
        else:
            if n == 17:
                 path_x = os.path.join(base_dir, f"n{n}_d{d}_Hx.alist")
                 path_z = os.path.join(base_dir, f"n{n}_d{d}_Hz.alist")
                 Hx = F2(readAlist(path_x))
                 Hz = F2(readAlist(path_z))
            else:
                raise FileNotFoundError(f"Missing separate files:\n  X: {path_x}\n  Z: {path_z}")

    return np.array(Hx, dtype=np.uint8), np.array(Hz, dtype=np.uint8)

def find_logical_operator(Hx, Hz, basis="Z"):
    F2 = galois.GF(2)
    gf_Hx = F2(Hx.astype(int))
    gf_Hz = F2(Hz.astype(int))
    candidates_basis = gf_Hx.null_space() if basis == "Z" else gf_Hz.null_space()
    stabilizers = gf_Hz if basis == "Z" else gf_Hx
    if not isinstance(stabilizers, F2):
        stabilizers = F2(stabilizers.astype(int))
    stab_rank = np.linalg.matrix_rank(stabilizers)
    for cand in candidates_basis:
        cand_row = np.atleast_2d(cand)
        combined = np.concatenate((stabilizers, cand_row), axis=0)
        if np.linalg.matrix_rank(combined) > stab_rank:
            return np.array(cand, dtype=np.uint8)
    if candidates_basis.shape[0] > 0:
        rows = candidates_basis
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                cand = rows[i] + rows[j]
                cand_row = np.atleast_2d(cand)
                combined = np.concatenate((stabilizers, cand_row), axis=0)
                if np.linalg.matrix_rank(combined) > stab_rank:
                    return np.array(cand, dtype=np.uint8)
    raise ValueError(f"Could not find a Logical {basis} operator!")

def parse_and_average_stats(stats: List[sinter.TaskStats], model_name: str) -> pd.DataFrame:

    results = []
    
    for s in stats:
        m = s.json_metadata
        trace_file = m.get('trace_path')
        
        avg_obj = 0.0
        avg_cpu = 0.0
        
        if trace_file and glob.glob(f"{trace_file}*"):
            try:
                df_details = SinterMWPFDecoder.parse_mwpf_trace(trace_file)
                if not df_details.empty:
                    avg_obj = df_details['objective_value'].mean()
                    avg_cpu = df_details['cpu_time'].mean()
            except Exception as e:
                # Log warning but continue
                pass

        results.append({
            'noise_model': model_name, 
            'n': m.get('n'), 
            'd': m.get('d'), 
            'r': m.get('r'), 
            'p': m.get('p'),
            'code_type': m.get('code_type', 'unknown'), 
            'shots': s.shots, 
            'errors': s.errors,
            'total_logical_error_rate': s.errors / s.shots if s.shots > 0 else 0,
            'mean_objective_value': avg_obj, 
            'average_cpu_time_seconds': avg_cpu,
        })
    return pd.DataFrame(results)