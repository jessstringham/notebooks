class Parent(object):
    '''Fancy accessor helper that ensures components can't see the future and
    helps return either the recent value or a full history.

    Parent.get_value is important for avoiding loops, while still being
    deterministic about which timestep an answer is gotten from.

    :param prev_timestep: If True, get_value will access the previous
      timestep.
    :param with_history: If True, returns a list of history. If used
      with prev_timestep, will return up until the previous timestep.
      Otherwise will return up until the current timestep.
    '''

    def __init__(
        self,
        component_name,
        prev_timestep=False,
        with_history=False,
    ):
        self.component_name = component_name
        self.prev_timestep = prev_timestep
        self.with_history = with_history

    def get_value(self, t, timestep_results):
        """Based on the Parent definition, access the history of
        the parent.

        :param t: current timestep
        :param timestep_results: full dictionary of all component's history

        :returns: float or list of floats
        """

        # Splice list up until `t + 1` to include the value at time step t
        last_timestep = t + 1
        if self.prev_timestep:
            last_timestep = last_timestep - 1

        time_scope = timestep_results[self.component_name][:last_timestep]

        # For nice function definitions, either return the last element, or return the full history.
        if self.with_history:
            return time_scope
        else:
            return time_scope[-1]


class Component(object):
    '''Base class for components. Shouldn't initiate this directly.'''
    _allowed_parents = []  # A list of Component types that can be a parent

    def reset(self):
        self.value = None if self.initial_value is None else float(self.initial_value)

    def __init__(
        self,
        name,
        equation=None,
        initial_value=None,
        parents=None,
    ):
        # save static attributes
        self.name = name
        self.initial_value = initial_value  # If simulation is reran
        self.equation = equation
        self.parents = parents or []

        # initialize
        self.reset()

    def __str__(self):
        return '{}'.format(self.name)

    def get_inputs(self, t, timestep_results):
        return {
            p.component_name: p.get_value(t, timestep_results)
            for p in self.parents
        }

    def _new_value(self, t, dt, prev_value, inputs):
        '''Should be implemented for each subtype. Calls the component's equation
        and returns the new value.'''
        raise NotImplementedError()

    def simulate_timestep(self, t, dt, timestep_results):
        inputs = self.get_inputs(t, timestep_results)
        # make sure that the new value is float
        self.value = float(self._new_value(t, dt, self.value, inputs))


class FlowComponent(Component):
    """A flow component."""
    _allowed_parents = ['InfoComponent', 'StockComponent']

    def _new_value(self, t, dt, prev_value, inputs):
        return self.equation(t, **inputs)


class InfoComponent(FlowComponent):
    """An information component passes information to the flows of a system."""
    _allowed_parents = ['StockComponent', 'FlowComponent', 'InfoComponent']


class StockComponent(Component):
    """A stock component. This represents something that grows and shrinks."""
    _allowed_parents = ['FlowComponent']

    def __init__(
        self,
        name,
        initial_value,
        inflow,
        outflow,
        min_value
    ):
        self.min_value = min_value
        self.inflow = inflow
        self.outflow = outflow

        super().__init__(
            name,
            equation=None,
            initial_value=initial_value,
            parents=[Parent(inflow, prev_timestep=True), Parent(outflow, prev_timestep=True)],
        )

    def _new_value(self, t, dt, prev_value, inputs):
        # The equation is always given by the inflow - outflow
        return max(self.min_value, prev_value + dt * (inputs[self.inflow] - inputs[self.outflow]))


class System(object):
    def __init__(self, components_list):
        self.components = {
            component.name: component
            for component in components_list
        }

        self._integrity_checks()

    def tree(self, with_prev_timestamp=True):
        return {
            node.name: [
                p.component_name
                for p in node.parents
                if with_prev_timestamp or not p.prev_timestep
            ]
            for node in self.components.values()
        }

    def _integrity_checks(self):
        # all parents are defined
        for component_name, component in self.components.items():
            for parent in component.parents:
                assertion_failed_str = "Component {}({}) parent `{}` not defined".format(
                    component.name,
                    type(component).__name__,
                    parent.component_name,
                )
                assert parent.component_name in self.components, assertion_failed_str

        # A time step shouldn't have loops. Also stores a topological sort.
        self.sorted_nodes = topological_sort(self.tree(with_prev_timestamp=False))

        # Make sure connections are allowed
        for component in self.components.values():
            for parent in component.parents:
                parent_type = type(self.components[parent.component_name]).__name__
                assertion_failed_str = "Component {}({}) can't have parent {}({})".format(
                    component.name,
                    type(component).__name__,
                    parent.component_name,
                    parent_type
                )
                assert parent_type in component._allowed_parents, assertion_failed_str

        # If something is a prev_parent, it needs an initial value.
        for component_name, component in self.components.items():
            for parent in component.parents:
                if parent.prev_timestep:
                    assertion_failed_str = "Component {} prev_parent `{}` needs initial value".format(
                        component.name,
                        parent.component_name,
                    )
                    assert self.components[parent.component_name].value is not None, assertion_failed_str

    def __str__(self):
        return ', '.join(map(str, self.components))

    def simulate(self, t_num, dt):
        # Clear history of components
        for c in self.components.values():
            c.reset()

        # Load initial values
        history = {
            name: [node.value]
            for name, node in self.components.items()
        }

        # Simulate for t_num timesteps
        for t in range(1, t_num + 1):
            for name in self.sorted_nodes:
                self.components[name].simulate_timestep(t, dt, history)
                # update with new values
                history[name].append(self.components[name].value)

        # Strip the initial value
        history = {
            name: timesteps[1:]
            for name, timesteps in history.items()
        }

        return history


def topological_sort(tree):
    '''Returns list of nodes in the provided tree in topological sorted.'''

    sorted_nodes = []

    # loop until tree is empty. But use a for-loop so don't loop forever:
    # an element should be removed each loop, so it shouldn't iterate more than
    # len(tree) + 1 times before exiting.
    for _ in range(len(tree) + 1):
        # Done if cleared out tree!
        if not tree:
            break

        # Find all of the nodes with no more parents
        for name in list(tree):
            if not tree[name]:
                sorted_nodes.append(name)
                del tree[name]

        # Clean out solved parents
        tree = {
            node_name: [p for p in parents if p in tree]
            for node_name, parents in tree.items()
        }
    else:
        # if nodes remain, something is wrong
        raise Exception("Not all nodes are reachable", tree)

    return sorted_nodes

