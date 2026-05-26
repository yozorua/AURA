"""Public API for the AURA data generation module."""

from .configs import (
    AtmosphericConfig,
    DatasetConfig,
    MountConfig,
    NoiseMode,
    SensorConfig,
    SpatialVarianceConfig,
    TelescopeConfig,
    TelescopeType,
)

# dataset.py requires torch — imported lazily so physics modules remain usable
# in torch-free environments (e.g. data-generation benchmarks, unit tests).
try:
    from .dataset import (
        AURASyntheticDataset,
        AURAValidationDataset,
        get_worker_init_fn,
    )
except ModuleNotFoundError:
    pass  # torch not installed; Dataset classes unavailable

from .psf_engine import (
    ApertureGenerator,
    AtmosphericSeeing,
    MountMechanics,
    PSFEngine,
    SensorModel,
    SpatialAberrationMap,
    ZernikePhaseScreen,
)

__all__ = [
    "DatasetConfig", "TelescopeConfig", "AtmosphericConfig", "MountConfig",
    "SensorConfig", "SpatialVarianceConfig", "TelescopeType", "NoiseMode",
    "AURASyntheticDataset", "AURAValidationDataset", "get_worker_init_fn",
    "PSFEngine", "ApertureGenerator", "ZernikePhaseScreen",
    "SpatialAberrationMap", "AtmosphericSeeing", "MountMechanics", "SensorModel",
]
