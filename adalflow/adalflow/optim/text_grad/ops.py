"""Text-grad operations such as Sum and Aggregate."""

from typing import List
import logging

from adalflow.optim.function import BackwardContext
from adalflow.optim.parameter import Parameter
from adalflow.optim.types import ParameterType
from adalflow.optim.grad_component import GradComponent

log = logging.getLogger(__name__)


def sum_ops(params: List[Parameter]) -> Parameter:
    """
    Represents a sum operation on a list of variables.
    In TextGrad, sum is simply concatenation of the values of the variables.

    :param variables: The list of variables to be summed (concatenated).
    :type variables: List[Variable]
    :return: A new variable representing the sum of the input variables.
    :rtype: Variable
    """
    for param in params:
        if not isinstance(param, Parameter):
            raise ValueError(
                f"Sum operation only accepts a list of Parameters, got {type(param)}"
            )
    return Sum()(params)


# TODO: there might be a better way to do this.
# TODO: make all loss functions to support batch losses
# TODO: use a temlate to format the concatenated values
class Sum(GradComponent):
    __doc__ = """The class to define a sum operation on a list of parameters, such as losses or gradients."""

    name = "Sum"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, params: List[Parameter]) -> Parameter:
        """
        Performs the forward pass of the sum operation.
        This is a simple operation that concatenates the values of the parameters.

        :param params: The list of parameters to be summed.
        :type params: List[Parameter]
        :rtype: Parameter
        """
        for param in params:
            if not isinstance(param, Parameter):
                raise ValueError(
                    f"Sum operation only accepts a list of Parameters, got {type(param)}"
                )
        concat_values = "\n".join([str(p.data) for p in params])  # to_dict
        role_descriptions = set([p.role_desc for p in params])
        role_descriptions = ", ".join(role_descriptions)

        total = Parameter(
            data=concat_values,
            role_desc=f"A combination of a list of variables: {role_descriptions}",
            requires_opt=any([p.requires_opt for p in params]),
            name="sum",
            score=sum([p._score for p in params]),  # total has a score
        )
        total.set_predecessors(params)

        log.info("Sum forward", extra={"total": total.data})

        total.set_grad_fn(BackwardContext(backward_fn=self.backward, summation=total))

        return total

    def backward(self, summation: Parameter):
        """
        Performs the backward pass of the sum operation.
        This is simply an idempotent operation, where we make a gradient with the combined feedback and add it to the predecessors'grads.

        :param summation: The parameter representing the sum.
        :type summation: Parameter
        """
        log.info(f"Sum backward: {summation.data}")
        pred_params = summation.predecessors  # losses
        summation_gradients = summation.get_gradient_and_context_text().strip()
        for param in pred_params:

            if param.check_if_already_computed_gradient_respect_to(summation.id):
                log.info(
                    f"Gradient already computed for {param.role_desc} with respect to {summation.role_desc}"
                )
                print(
                    f"Gradient already computed for {param.role_desc} with respect to {summation.role_desc}"
                )
                continue

            # add a combined gradients
            if (
                summation_gradients == ""
            ):  # as loss sum to be the base, it simply allows gradients computations on multiple losses
                param_gradient_value = ""
            else:  # as a mid layer, it will have a combined feedback

                param_gradient_value = f"Here is the combined feedback we got for this specific {param.role_desc} and other parameters: {summation_gradients}."

            extra = {
                "p_gradient_value": param_gradient_value,
                "summation_role": summation.role_desc,
            }
            log.info(f"""Idempotent sum backward: {extra}""")

            param_gradient = Parameter(
                name=f"sum_to_{param.name}_grad",
                data=param_gradient_value,
                role_desc=f"Feedback to {param.role_desc}",
                score=summation._score,
                from_response_id=summation.id,
                param_type=ParameterType.GRADIENT,
            )
            param.add_gradient(param_gradient)
            log.debug(f"Added gradient to {param.role_desc}: {param_gradient.data}")
