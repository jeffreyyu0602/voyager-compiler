"""
Top level function of optimization framework
"""

import logging

from . import mapping_point_generator
from . import cost_model

from . import loop_enum as le
from . import buffer_enum as be

logger = logging.getLogger(__name__)


def opt_optimizer(resource, layer, hint=None, runtime_calc_func=None, verbose=False):
    """
    Evaluate the cost of each mapping point,
    record the mapping_point with the smallest cost
    """

    smallest_cost, smallest_runtime, perf, best_mapping_point = (
        mapping_point_generator.opt_mapping_point_generator_function(
            resource, layer, hint, runtime_calc_func, verbose
        )
    )
    access_list, array_cost = cost_model.get_access(
        best_mapping_point, layer, resource
    )
    logger.info("Access_list: %s", access_list)
    logger.info("Array_cost: %s", array_cost)

    logger.debug("Optimal_Energy_(pJ): %.2e", smallest_cost)
    logger.debug("Runtime_(cycles): %s", perf)

    return [smallest_cost, smallest_runtime, best_mapping_point, perf]


def optimizer(resource, layer, hint=None):
    smallest_cost = float("inf")
    mp_generator = mapping_point_generator.mapping_point_generator_function(
        resource, layer, hint
    )

    for mapping_point in mp_generator:
        cost = cost_model.get_cost(resource, mapping_point, layer)

        if cost < smallest_cost:
            smallest_cost = cost
            best_mapping_point = mapping_point
            logger.debug("Current smallest cost: %s", smallest_cost)
            logger.debug(
                "Current best mapping_point: %s %s",
                mapping_point.loop_blockings,
                mapping_point.loop_orders,
            )

    logger.debug("Smallest cost: %s", smallest_cost)

    return [smallest_cost, best_mapping_point]
