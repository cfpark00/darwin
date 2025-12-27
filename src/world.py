"""World generation - backwards compatibility re-exports.

All world generation code has moved to src/worlds/.
This file re-exports for backwards compatibility.
"""

from src.worlds import create_world, regenerate_resources
from src.worlds.base import (
    generate_gaussian_field,
    generate_resource_field,
    generate_temperature_field,
    generate_toxin_field,
)

__all__ = [
    "create_world",
    "regenerate_resources",
    "generate_gaussian_field",
    "generate_resource_field",
    "generate_temperature_field",
    "generate_toxin_field",
]
