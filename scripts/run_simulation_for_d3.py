'''
Script for caching three graphs from one of the examples from the post [1],
for use in an interactive post.

[1] https://jessicastringham.net/2019/07/01/systems-modeling-from-scratch/
'''
import numpy as np
import json


from model_simulation import *


# In this model, fish regenerate slower if there aren't many other fish, or if there are too many other fish.
def regeneration_rate_given_resource(resource):
    scaled_resource = (resource/1000)

    if scaled_resource < 0.5:
        adjusted_resource = scaled_resource
    else:
        adjusted_resource = (1 - scaled_resource)

    rate = np.tanh(12 * adjusted_resource - 3)
    rate = (rate + 1)/4
    return max(0, rate)


# People require fish, and are willing to pay more for fish if it is scarce.
def price_given_yield_per_unit_capital(yield_per_unit_capital):
    return 8.8 * np.exp(-yield_per_unit_capital*4) + 1.2


def yield_per_unit_capital_given_resource(resource, some_measure_of_efficiency):
    return min(1, max(0, (np.tanh(resource/1000*6 - 3 + some_measure_of_efficiency))/1.9 + 0.5))


def renewable_resource(some_measure_of_efficiency):
    return System([
        StockComponent(
            name='resource',
            initial_value=1000,
            inflow='regeneration',
            outflow='harvest',
            min_value=0,
        ),
        FlowComponent(
            name='regeneration',
            initial_value=0,
            equation=lambda t, resource, regeneration_rate: resource * regeneration_rate,
            parents=[Parent('resource'), Parent('regeneration_rate')]
        ),
        FlowComponent(
            name='harvest',
            initial_value=0,
            equation=lambda t, resource, capital, yield_per_unit_capital: min(resource, capital * yield_per_unit_capital),
            parents=[Parent('resource'), Parent('capital', prev_timestep=True), Parent('yield_per_unit_capital')]
        ),

        StockComponent(
            name='capital',
            initial_value=5,
            inflow='investment',
            outflow='depreciation',
            min_value=0,
        ),
        FlowComponent(
            name='investment',
            equation=lambda t, profit, growth_goal: max(0, min(profit, growth_goal)),
            initial_value=0,
            parents=[Parent('profit'), Parent('growth_goal')]
        ),
        FlowComponent(
            name='depreciation',
            equation=lambda t, capital, capital_lifetime: capital/capital_lifetime,
            initial_value=0,
            parents=[Parent('capital', prev_timestep=True), Parent('capital_lifetime')]
        ),

        InfoComponent(
            name='capital_lifetime',
            equation=lambda _: 20),

        InfoComponent(
            name='growth_goal',
            equation=lambda t, capital: capital * .1,
            parents=[Parent('capital', prev_timestep=True)]),

        InfoComponent(
            name='profit',
            equation=lambda t, price, harvest, capital: (price * harvest) - capital,
            parents=[Parent('price'), Parent('harvest'), Parent('capital', prev_timestep=True)]
        ),
        InfoComponent(
            name='price',
            equation=lambda t, yield_per_unit_capital: price_given_yield_per_unit_capital(yield_per_unit_capital),
            parents=[Parent('yield_per_unit_capital')]
        ),
        InfoComponent(
            name='regeneration_rate',
            equation=lambda t, resource: regeneration_rate_given_resource(resource),
            parents=[Parent('resource')]),
        InfoComponent(
            name='yield_per_unit_capital',
            equation=lambda t, resource: yield_per_unit_capital_given_resource(resource, some_measure_of_efficiency),
            parents=[Parent('resource')]
        )
    ])


if __name__ == '__main__':
    years = 200
    samples_per_year = 1
    t_num = years * samples_per_year
    dt = 1/samples_per_year

    # First generate all data. For each some_measure_of_efficiency,
    # compute the three graphs.
    xs_yield = np.linspace(0, 1000, 100)
    xs_simulated_resource = np.linspace(0, years, t_num + 1)
    xs_simulated_capital = xs_simulated_resource  # same as resource

    graph_data_yield = {}
    graph_data_simulated_resource = {}
    graph_data_simulated_capital = {}

    tap_location_values = np.linspace(-0.5, 1.5, 20)

    # Cache the function for each tap_locations
    for yield_parameter in tap_location_values:
        s = renewable_resource(yield_parameter)
        simulation = s.simulate(t_num, dt)

        graph_data_yield[yield_parameter] = [
            yield_per_unit_capital_given_resource(x, yield_parameter)
            for x in xs_yield
        ]

        graph_data_simulated_resource[yield_parameter] = simulation['resource']
        graph_data_simulated_capital[yield_parameter] = simulation['capital']


    database = {
        'yield_parameters': list(sorted(graph_data_yield.keys())),
        'yield_graph': {
            'xs': list(xs_yield),
            'ys_by_yield_parameter': graph_data_yield,
            'x_domain': [min(xs_yield), max(xs_yield)],
            'y_domain': [0, 1],
        },
        'yield_simulated_resource': {
            'xs': list(xs_simulated_resource),
            'ys_by_yield_parameter': graph_data_simulated_resource,
            'x_domain': [min(xs_simulated_resource), max(xs_simulated_resource)],
            'y_domain': [0, 1200],
        },
        'yield_simulated_capital': {
            'xs': list(xs_simulated_capital),
            'ys_by_yield_parameter': graph_data_simulated_capital,
            'x_domain': [min(xs_simulated_capital), max(xs_simulated_capital)],
            'y_domain': [0, 1200],
        },
    }

    print(json.dumps(database))

