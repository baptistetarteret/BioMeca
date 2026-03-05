import numpy as np

def compute_efficiency_2d(mesh_nodes, node_tissue, snapshots, T_injury=46.0, T_lethal=60.0, tumor_name="tumor", healthy_name="brain"):
    T_injury = float(T_injury)  
    T_lethal = float(T_lethal) 

    x = mesh_nodes[:, 0]  
    y = mesh_nodes[:, 1]  
    r = np.sqrt(x * x + y * y)  

    w = r + 1e-12  

    tissue_names = np.array([t.name for t in node_tissue], dtype=object)  

    T_stack = np.vstack(snapshots)  
    Tmax = np.max(T_stack, axis=0) 

    is_tumor = (tissue_names == tumor_name) 
    is_healthy = (tissue_names == healthy_name)  

    Area_tumor = float(np.sum(w[is_tumor]))  
    Area_healthy = float(np.sum(w[is_healthy]))  

    Area_tumor_lethal = float(np.sum(w[is_tumor & (Tmax >= T_lethal)]))  

    Area_healthy_injury = float(np.sum(w[is_healthy & (Tmax >= T_injury)]))  
    Area_healthy_lethal = float(np.sum(w[is_healthy & (Tmax >= T_lethal)]))  

    frac_tumor_lethal = Area_tumor_lethal / max(Area_tumor, 1e-12)  # Fraction of tumor area (weighted) that reached ≥ 60°C
    frac_healthy_injury = Area_healthy_injury / max(Area_healthy, 1e-12) # Fraction of brain area that reached ≥ 46°C  
    frac_healthy_lethal = A_healthy_lethal / max(Area_healthy, 1e-12) 

    score = frac_tumor_lethal - 0.5 * frac_healthy_injury - 1.0 * frac_healthy_lethal 

    return {
        "score": float(score),  
        "Area_tumor_w": Area_tumor,  
        "Area_healthy_w": Area_healthy, 
        "Area_tumor_lethal_w": Area_tumor_lethal,  
        "Area_healthy_injury_w": Area_healthy_injury,  
        "Area_healthy_lethal_w": Area_healthy_lethal,  
        "frac_tumor_lethal": float(frac_tumor_lethal),  
        "frac_healthy_injury": float(frac_healthy_injury),  
        "frac_healthy_lethal": float(frac_healthy_lethal),  
        "T_injury": float(T_injury),  
        "T_lethal": float(T_lethal),  
        "tumor_name": tumor_name,  
        "healthy_name": healthy_name,  
    }
