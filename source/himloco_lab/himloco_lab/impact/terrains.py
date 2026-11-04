"""Factorized A0/A1 terrain suite. Import after SimulationApp starts.

Generator curriculum is used ONLY to assign known families to columns and
stratify difficulty. Environment curriculum remains disabled throughout.
"""
import isaaclab.terrains as terrain

FAMILIES = ("plane", "rough", "slope_down", "slope_up", "stairs_down", "stairs_up", "boxes_heldout")


def research_terrain(seed):
    common = dict(proportion=1.0, convert_to_heightfield=True)
    return terrain.TerrainGeneratorCfg(
        seed=seed, size=(12.0, 12.0), border_width=10.0,
        num_rows=6, num_cols=14, curriculum=True, difficulty_range=(0.0, 1.0),
        horizontal_scale=0.1, vertical_scale=0.0025, slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "plane": terrain.MeshPlaneTerrainCfg(**common),
            "rough": terrain.HfRandomUniformTerrainCfg(
                **common, noise_range=(0.01, 0.04), noise_step=0.005,
                downsampled_scale=0.2, border_width=0.25),
            "slope_down": terrain.HfPyramidSlopedTerrainCfg(
                **common, slope_range=(0.05, 0.25), platform_width=1.5, border_width=0.25),
            "slope_up": terrain.HfInvertedPyramidSlopedTerrainCfg(
                **common, slope_range=(0.05, 0.25), platform_width=1.5, border_width=0.25),
            "stairs_down": terrain.MeshPyramidStairsTerrainCfg(
                **common, step_height_range=(0.02, 0.10), step_width=0.30,
                platform_width=1.5, border_width=0.5, holes=False),
            "stairs_up": terrain.MeshInvertedPyramidStairsTerrainCfg(
                **common, step_height_range=(0.02, 0.10), step_width=0.30,
                platform_width=1.5, border_width=0.5, holes=False),
            "boxes_heldout": terrain.MeshRandomGridTerrainCfg(
                **common, grid_width=0.45, grid_height_range=(0.02, 0.10), platform_width=1.5),
        })
