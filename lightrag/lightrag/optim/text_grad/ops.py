"""Text-grad operations such as Sum and Aggregate."""

from typing import List
import logging

from lightrag.optim.text_grad.function import GradFunction, BackwardContext
from lightrag.optim.parameter import Parameter

log = logging.getLogger(__name__)


def sum(parms: List[Parameter]) -> Parameter:
    """
    Represents a sum operation on a list of variables.
    In TextGrad, sum is simply concatenation of the values of the variables.

    :param variables: The list of variables to be summed (concatenated).
    :type variables: List[Variable]
    :return: A new variable representing the sum of the input variables.
    :rtype: Variable
    """
    return Sum()(parms)


# TODO: there might be a better way to do this.
# TODO: make all loss functions to support batch losses
# TODO: use a temlate to format the concatenated values
class Sum(GradFunction):
    __doc__ = """The class to define a sum operation on a list of parameters, such as losses or gradients."""

    def forward(self, params: List[Parameter]) -> Parameter:
        """
        Performs the forward pass of the sum operation.
        This is a simple operation that concatenates the values of the parameters.

        :param params: The list of parameters to be summed.
        :type params: List[Parameter]
        :rtype: Parameter
        """
        concat_values = "\n".join([str(p.data) for p in params])  # to_dict
        role_descriptions = set([p.role_desc for p in params])
        role_descriptions = ", ".join(role_descriptions)

        total = Parameter(
            data=concat_values,
            predecessors=params,
            role_desc=f"A combination of a list of variables: {role_descriptions}",
            requires_opt=any([p.requires_opt for p in params]),
        )

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
        pred_params = summation.predecessors  # losses
        summation_gradients = summation.get_gradient_text()
        for param in pred_params:

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
                alias=f"sum_to_{param.alias}_grad",
                data=param_gradient_value,
                role_desc=f"Feedback to {param.role_desc}",
            )
            param.gradients.add(param_gradient)
            log.debug(f"Added gradient to {param.role_desc}: {param_gradient.data}")
