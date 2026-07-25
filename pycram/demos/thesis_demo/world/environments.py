from thesis_demo.world.apartment import APARTMENT
from thesis_demo.world.kitchen import KITCHEN

ENVIRONMENTS = {
    "apartment": APARTMENT,
    "kitchen": KITCHEN,
}

FURNITURE_ANNOTATION_TYPES = {
    furniture.annotation_type
    for environment in ENVIRONMENTS.values()
    for furniture in environment.furniture
}
